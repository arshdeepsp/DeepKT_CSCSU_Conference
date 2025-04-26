import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gc

# -----------------------------------------------------------
# Helper functions for cloning modules (parameter sharing)
# -----------------------------------------------------------

def clone_once(module):
    """
    Clone a module such that the clone shares parameters with the original.
    """
    clone = copy.deepcopy(module)
    # Manually make parameters share storage.
    for orig_param, clone_param in zip(module.parameters(), clone.parameters()):
        clone_param.data = orig_param.data
    for orig_buf, clone_buf in zip(module.buffers(), clone.buffers()):
        clone_buf.data = orig_buf.data
    return clone

def clone_many_times(module, T):
    """
    Clone a module T times, with shared parameters.
    """
    clones = []
    for t in range(T):
        clones.append(clone_once(module))
        # print('clone', t + 1)
    return clones

# -----------------------------------------------------------
# Utility: get_n_steps (from utilExp.lua)
# -----------------------------------------------------------

def get_n_steps(batch):
    """
    Given a batch (a list of student answer dictionaries), return
    the maximum number of steps (i.e. answers) among students minus one.
    (Assumes each dict has key 'n_answers'.)
    """
    max_steps = 0
    for ans in batch:
        if ans['n_answers'] > max_steps:
            max_steps = ans['n_answers']
    return max_steps - 1

# -----------------------------------------------------------
# RNNLayer: a single recurrence unit
# -----------------------------------------------------------

class RNNLayer(nn.Module):
    def __init__(self, n_input, n_hidden, n_questions, dropout_pred):
        """
        The recurrence layer takes four inputs:
          - state (memory input) of shape (batch, n_hidden)
          - inputX (last student activity) of shape (batch, n_input)
          - inputY (one-hot vector for the next question answered) of shape (batch, n_questions)
          - truth (binary target for next answer) of shape (batch,)
        It returns a tuple: (pred, err, hidden) where:
          - pred: scalar prediction per sample (selected from the sigmoid output)
          - err: binary cross-entropy loss computed on pred and truth
          - hidden: the new hidden state (of shape (batch, n_hidden))
        """
        super(RNNLayer, self).__init__()
        # Transfer for the memory input (state)
        self.transfer = nn.Linear(n_hidden, n_hidden)
        # Process current inputX
        self.linear_x = nn.Linear(n_input, n_hidden)
        # Produce prediction from hidden state
        self.linear_y = nn.Linear(n_hidden, n_questions)
        self.dropout_pred = dropout_pred
        self.dropout = nn.Dropout() if dropout_pred else None
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, inputX, inputY, truth):
        """
        Forward pass of one time step.
        """
        # Process memory and current input separately.
        linM = self.transfer(state)          # shape: (batch, n_hidden)
        linX = self.linear_x(inputX)         # shape: (batch, n_hidden)
        hidden = self.activation(linM + linX)  # combine and apply nonlinearity
        # Optionally apply dropout to the hidden state before prediction.
        pred_input = self.dropout(hidden) if self.dropout is not None else hidden
        linY = self.linear_y(pred_input)     # shape: (batch, n_questions)
        pred_output = self.sigmoid(linY)       # elementwise sigmoid
        # Multiply elementwise with inputY (assumed one‐hot) and sum over questions
        pred = (pred_output * inputY).sum(dim=1)  # shape: (batch,)
        # Compute binary cross-entropy loss.
        # (Note: in Lua the BCECriterion returns a single-element tensor.)
        err = F.binary_cross_entropy(pred, truth, reduction='sum')
        return pred, err, hidden

# -----------------------------------------------------------
# Main RNN class
# -----------------------------------------------------------

class RNN:
    def __init__(self, params):
        """
        Initializes the RNN.
          params: a dictionary with keys:
            - 'n_questions': number of questions.
            - 'n_hidden': size of the hidden state.
            - 'dropout': (optional) whether to use dropout.
            - 'maxGrad': maximum gradient norm.
            - 'dropoutPred': whether to use dropout before prediction.
            - 'maxSteps': maximum number of time steps.
            - 'compressedSensing': (optional) if True, use compressed sensing.
            - 'compressedDim': (if compressedSensing) dimension after compression.
            - 'modelDir': (optional) directory from which to load a saved model.
        """
        self.n_questions = params['n_questions']
        self.n_hidden = params['n_hidden']
        self.use_dropout = params.get('dropout', False)
        self.max_grad = params['maxGrad']
        self.dropoutPred = params.get('dropoutPred', False)
        self.max_steps = params['maxSteps']
        # TODO: Retrieve the question mapping passed in params. Hongmin
        self.question_mapping = params.get('question_mapping')

        # The base input dimension is twice the number of questions.
        self.n_input = self.n_questions * 2
        self.compressed_sensing = params.get('compressedSensing', False)
        if self.compressed_sensing:
            self.n_input = params['compressedDim']
            torch.manual_seed(12345)
            # Create a random projection matrix (basis)
            self.basis = torch.randn(self.n_questions * 2, self.n_input)
        # Load a model if a directory is provided; otherwise build a new model.
        if 'modelDir' in params and params['modelDir'] is not None:
            self.load(params['modelDir'])
        else:
            self.build(params)
        print('rnn made')

    def build(self, params):
        """
        Build the network architecture.
          - self.start: a linear layer mapping from a scalar input (dummy zero) to hidden state.
          - self.layer: the recurrence unit.
          - Roll out the recurrence for self.max_steps time steps.
        """
        # "start" layer: from a dummy input of shape (batch, 1) to the initial hidden state.
        self.start = nn.Linear(1, self.n_hidden)
        # The recurrent layer that processes one time step.
        self.layer = RNNLayer(self.n_input, self.n_hidden, self.n_questions, self.dropoutPred)
        self.roll_out_network()

    def roll_out_network(self):
        """
        Clones the recurrence layer for each time step.
        (The clones share weights with self.layer.)
        """
        self.layers = clone_many_times(self.layer, self.max_steps)
        for layer in self.layers:
            layer.train()

    def zero_grad(self, n_steps):
        """
        Zero the gradients in the start and layer modules.
        """
        self.start.zero_grad()
        self.layer.zero_grad()

    def update(self, n_steps, rate):
        """
        A manual parameter update (gradient descent step) on the start and layer modules.
        (In typical PyTorch usage you would use an optimizer.)
        """
        for param in self.start.parameters():
            if param.grad is not None:
                param.data.add_(-rate, param.grad.data)
        for param in self.layer.parameters():
            if param.grad is not None:
                param.data.add_(-rate, param.grad.data)

    # def fprop(self, batch):
    #     """
    #     Forward propagate through time.
    #       batch: list of student answer dicts.
    #     Returns a tuple: (sum_err, num_tests, inputs)
    #       - sum_err: cumulative error (loss) scaled by the number of students.
    #       - num_tests: total number of test points.
    #       - inputs: list of inputs (per time step) used for the forward pass.
    #     """
    #     n_steps = get_n_steps(batch)
    #     n_students = len(batch)
    #     assert n_steps >= 1
    #     assert n_steps < self.max_steps
    #     inputs = []
    #     sum_err = 0.0
    #     num_tests = 0.0
    #     # Compute initial state from the "start" module.
    #     state = self.start(torch.zeros(n_students, 1))
    #     for k in range(1, n_steps + 1):
    #         inputX, inputY, truth = self.get_inputs(batch, k)
    #         mask = self.get_mask(batch, k)
    #         inputs.append((state, inputX, inputY, truth))
    #         # Use the k-th clone (index k-1) for this time step.
    #         pred, err, hidden = self.layers[k - 1](state, inputX, inputY, truth)
    #         state = hidden  # update the state for the next time step
    #         # In the Lua code, the error scalar is multiplied by the number of students.
    #         step_err = err.item() * n_students
    #         num_tests += mask.sum().item()
    #         sum_err += step_err
    #     return sum_err, num_tests, inputs
    def fprop(self, batch):
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        assert n_steps >= 1
        assert n_steps < self.max_steps
        inputs = []
        sum_err = 0.0
        num_tests = 0.0
        # Compute initial state and retain its grad.
        state = self.start(torch.zeros(n_students, 1))
        state.retain_grad()
        for k in range(1, n_steps + 1):
            inputX, inputY, truth = self.get_inputs(batch, k)
            inputs.append((state, inputX, inputY, truth))
            pred, err, hidden = self.layers[k - 1](state, inputX, inputY, truth)
            state = hidden  # update state
            state.retain_grad()  # ensure the new state will have its grad retained
            step_err = err.item() * n_students
            num_tests += self.get_mask(batch, k).sum().item()
            sum_err += step_err
        return sum_err, num_tests, inputs


    # def calc_grad(self, batch, rate, alpha):
    #     """
    #     Perform forward propagation and then (manually) backpropagate through time.
    #     (Note: In typical PyTorch code you would sum a loss over time and call backward() once.)
    #     Returns (sum_err, num_tests, max_norm).
    #     """
    #     n_steps = get_n_steps(batch)
    #     n_students = len(batch)
    #     assert n_steps <= self.max_steps

    #     max_norm = 0.0
    #     sum_err, num_tests, inputs = self.fprop(batch)

    #     parent_grad = torch.zeros(n_students, self.n_hidden)
    #     # Backpropagate through time (from the last time step to the first).
    #     for k in reversed(range(1, n_steps + 1)):
    #         state, inputX, inputY, truth = inputs[k - 1]
    #         layer = self.layers[k - 1]
    #         # Forward pass for this time step (to obtain outputs).
    #         pred, err, hidden = layer(state, inputX, inputY, truth)
    #         # Create a dummy gradient for the error term.
    #         grad_err = torch.ones(n_students) * alpha
    #         # TODO Hongmin commented our the following line
    #         # Backward pass for this time step.
    #         # err.backward(gradient=grad_err, retain_graph=True)
    #         (err * alpha).backward(retain_graph=True)
    #         # In the Lua code, parentGrad is taken as the gradient w.r.t. the state.
    #         if state.grad is not None:
    #             parent_grad = state.grad.clone()
    #         else:
    #             parent_grad = torch.zeros(n_students, self.n_hidden)
    #         state.grad.zero_()
    #     # Backpropagate through the "start" module.
    #     self.start(torch.zeros(n_students, 1)).backward(parent_grad)
    #     return sum_err, num_tests, max_norm
    def calc_grad(self, batch, rate, alpha):
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        assert n_steps <= self.max_steps

        max_norm = 0.0
        sum_err, num_tests, inputs = self.fprop(batch)

        parent_grad = None
        # Backpropagate through time in reverse.
        for k in reversed(range(1, n_steps + 1)):
            state, inputX, inputY, truth = inputs[k - 1]
            layer = self.layers[k - 1]
            pred, err, hidden = layer(state, inputX, inputY, truth)
            # Scale the loss by alpha and backward.
            (err * alpha).backward(retain_graph=True)
            # Retrieve gradient from state if available.
            if state.grad is not None:
                parent_grad = state.grad.clone()
                state.grad.zero_()  # zero out for the next backward pass
            else:
                parent_grad = torch.zeros(n_students, self.n_hidden, device=state.device)
        self.start(torch.zeros(n_students, 1)).backward(parent_grad)
        return sum_err, num_tests, max_norm


    def get_mask(self, batch, k):
        """
        For the given time step k (1-indexed) return a mask (tensor of shape (n_students,))
        with 1 where the student has a (k+1)-th answer and 0 otherwise.
        """
        n = len(batch)
        mask = torch.zeros(n)
        for i, ans in enumerate(batch):
            if k + 1 <= ans['n_answers']:
                mask[i] = 1
        return mask

    # def get_inputs(self, batch, k):
    #     """
    #     For time step k (1-indexed) extract:
    #       - inputX: a (n_students x (2*n_questions)) tensor (one-hot encoded based on current activity)
    #       - inputY: a (n_students x n_questions) tensor (one-hot for the next question answered)
    #       - truth: a (n_students,) tensor (the correctness of the next answer)
    #     (Note: This function assumes that in each student record, the lists for 'questionId'
    #      and 'correct' are 1-indexed, so we subtract 1 for Python indexing.)
    #     """
    #     n_students = len(batch)
    #     mask = self.get_mask(batch, k)
    #     inputX = torch.zeros(n_students, 2 * self.n_questions)
    #     inputY = torch.zeros(n_students, self.n_questions)
    #     truth = torch.zeros(n_students)
    #     for i, answers in enumerate(batch):
    #         if k + 1 <= answers['n_answers']:
    #             # Lua indices are 1-indexed; here we assume the stored lists use 1-indexing.
    #             currentId = answers['questionId'][k - 1]
    #             nextId = answers['questionId'][k]
    #             currentCorrect = answers['correct'][k - 1]
    #             nextCorrect = answers['correct'][k]
    #             xIndex = self.get_x_index(currentCorrect, currentId)
    #             # Adjust xIndex from 1-indexed to 0-indexed.
    #             inputX[i, xIndex - 1] = 1
    #             truth[i] = nextCorrect
    #             # For inputY, assume nextId is 1-indexed.
    #             inputY[i, nextId - 1] = 1
    #     # If compressed sensing is enabled, project inputX using the random basis.
    #     if self.compressed_sensing:
    #         inputX = inputX @ self.basis
    #     return inputX, inputY, truth
    def get_inputs(self, batch, k):
        n_students = len(batch)
        mask = self.get_mask(batch, k)
        inputX = torch.zeros(n_students, 2 * self.n_questions)
        inputY = torch.zeros(n_students, self.n_questions)
        truth = torch.zeros(n_students)
        for i, answers in enumerate(batch):
            if k + 1 <= answers['n_answers']:
                raw_currentId = answers['questionId'][k - 1]
                raw_nextId = answers['questionId'][k]
                currentCorrect = answers['correct'][k - 1]
                nextCorrect = answers['correct'][k]
                
                # Use the mapping from dataAssist
                currentId = self.question_mapping.get(raw_currentId, None)
                nextId = self.question_mapping.get(raw_nextId, None)
                if currentId is None or nextId is None:
                    raise ValueError("Question id not found in mapping")
                
                xIndex = self.get_x_index(currentCorrect, currentId)
                inputX[i, xIndex - 1] = 1
                
                truth[i] = nextCorrect
                inputY[i, nextId - 1] = 1
        if self.compressed_sensing:
            inputX = inputX @ self.basis
        return inputX, inputY, truth


    def get_x_index(self, correct, id):
        """
        Compute the index into inputX based on whether the answer was correct.
        In Lua: xIndex = correct * n_questions + id (with 1 <= xIndex <= 2*n_questions)
        """
        assert correct in (0, 1)
        assert id != 0
        x_index = correct * self.n_questions + id
        assert 1 <= x_index <= 2 * self.n_questions
        return x_index

    def clone_many_times(self, n):
        """
        Clone the recurrence layer n times (sharing parameters).
        """
        return clone_many_times(self.layer, n)

    def prevent_explosion(self, grad):
        """
        If the norm of the gradient exceeds self.max_grad, scale it down.
        """
        norm = grad.norm()
        if norm > self.max_grad:
            print('explosion')
            alpha = self.max_grad / norm
            grad.mul_(alpha)
        return norm

    def err(self, batch):
        """
        Evaluate the error (loss) over a batch.
        Switches the layers to evaluation mode, performs forward propagation,
        then resets them to training mode.
        Returns the average error per test.
        """
        n_steps = get_n_steps(batch)
        for i in range(n_steps):
            self.layers[i].eval()
        sum_err, num_tests, inputs = self.fprop(batch)
        for i in range(n_steps):
            self.layers[i].train()
        return sum_err / num_tests if num_tests != 0 else 0

    def accuracy(self, batch):
        """
        Compute the accuracy over a batch.
        Returns (sum_correct, num_tested).
        """
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        self.layer.eval()
        sum_correct = 0.0
        num_tested = 0.0
        state = self.start(torch.zeros(n_students, 1))
        for k in range(1, n_steps + 1):
            inputX, inputY, truth = self.get_inputs(batch, k)
            pred, _, hidden = self.layer(state, inputX, inputY, truth)
            state = hidden.clone()
            mask = self.get_mask(batch, k)
            p = pred.double()
            # Binary prediction using a threshold of 0.5.
            pred_bin = (p > 0.5).double()
            correct = (pred_bin == truth).double()
            num_correct = (correct * mask).sum().item()
            sum_correct += num_correct
            num_tested += mask.sum().item()
        # Reset to training mode (for dropout, etc.)
        for i in range(n_steps):
            self.layer.train()
        return sum_correct, num_tested

    def get_prediction_truth(self, batch):
        """
        Runs the model on the batch (with dropout disabled) and returns a list of
        dictionaries, each containing the predicted value and the ground truth,
        for every test point.
        """
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        self.layer.eval()
        prediction_truths = []
        state = self.start(torch.zeros(n_students, 1))
        for k in range(1, n_steps + 1):
            inputX, inputY, truth = self.get_inputs(batch, k)
            pred, _, hidden = self.layer(state, inputX, inputY, truth)
            state = hidden.clone()
            mask = self.get_mask(batch, k)
            p = pred.double()
            for i in range(n_students):
                if mask[i] == 1:
                    prediction_truths.append({'pred': p[i].item(), 'truth': truth[i].item()})
        self.layer.train()
        return prediction_truths

    def save(self, dir):
        """
        Save the start and recurrence layer modules.
        """
        os.makedirs(dir, exist_ok=True)
        torch.save(self.start, os.path.join(dir, 'start.dat'))
        torch.save(self.layer, os.path.join(dir, 'layer.dat'))

    def load(self, dir):
        """
        Load the start and recurrence layer modules.
        """
        self.start = torch.load(os.path.join(dir, 'start.dat'))
        self.layer = torch.load(os.path.join(dir, 'layer.dat'))
        self.roll_out_network()
