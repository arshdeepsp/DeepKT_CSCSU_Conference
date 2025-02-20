import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from data_assist import DataAssistMatrix
from util_exp import semi_sorted_mini_batches
from torch.utils.data import Dataset, DataLoader


class DKTCell(nn.Module):
    """
    Implements one time step of the DKT model.
    Computes:
      hidden = tanh( transfer(state) + linear_x(inputX) )
      (optionally applies dropout)
      pred_output = sigmoid( linear_y(hidden) )
      pred = sum( pred_output * inputY, dim=1 )
    And computes binary cross-entropy loss between pred and truth.
    """
    def __init__(self, n_input, n_hidden, n_questions, dropout_pred=False):
        super(DKTCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_questions = n_questions
        self.dropout_pred = dropout_pred

        self.transfer = nn.Linear(n_hidden, n_hidden)
        self.linear_x = nn.Linear(n_input, n_hidden)
        self.linear_y = nn.Linear(n_hidden, n_questions)
        self.dropout = nn.Dropout() if dropout_pred else None

    def forward(self, state, inputX, inputY, truth):
        """
        state: Tensor (batch, n_hidden)
        inputX: Tensor (batch, n_input)
        inputY: Tensor (batch, n_questions) – one-hot for next question.
        truth:  Tensor (batch,) – binary correctness.
        Returns:
           pred: Tensor (batch,) – predicted probability (scalar per sample)
           loss: scalar loss (sum over batch)
           hidden: new hidden state (batch, n_hidden)
        """
        hidden = torch.tanh(self.transfer(state) + self.linear_x(inputX))
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        linY = self.linear_y(hidden)
        pred_output = torch.sigmoid(linY)
        # Multiply elementwise with inputY and sum over question dimension.
        pred = torch.sum(pred_output * inputY, dim=1)
        loss = F.binary_cross_entropy(pred, truth, reduction='sum')
        return pred, loss, hidden


class DKT(nn.Module):
    """
    Deep Knowledge Tracing model.
    """
    def __init__(self, n_questions, n_hidden, dropout_pred=False,
                 compressed_sensing=False, compressed_dim=None,
                 question_mapping=None):
        super(DKT, self).__init__()
        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.dropout_pred = dropout_pred
        self.compressed_sensing = compressed_sensing
        self.question_mapping = question_mapping  # e.g., { raw_qid: contiguous_index }

        if self.compressed_sensing:
            assert compressed_dim is not None, "compressed_dim must be specified"
            self.n_input = compressed_dim
            # Create a random projection matrix from (2*n_questions) to compressed_dim.
            torch.manual_seed(12345)
            # Use a non-trainable parameter for the projection basis.
            self.basis = nn.Parameter(torch.randn(n_questions * 2, self.n_input), requires_grad=False)
        else:
            self.n_input = n_questions * 2
            self.basis = None

        self.cell = DKTCell(self.n_input, n_hidden, n_questions, dropout_pred)
        self.start_layer = nn.Linear(1, n_hidden)

    def forward(self, batch):
        """
        Forward pass on a batch of student records.
        Each record is a dict with:
           - 'n_answers': int, number of answers
           - 'questionId': list of raw question IDs (assumed 1-indexed, or remapped via question_mapping)
           - 'correct': list of binary correctness (0 or 1)
        The model runs for T = max(n_answers) - 1 time steps.
        Returns:
          total_loss: scalar loss (sum over time and students)
          total_tests: total number of prediction points (for averaging)
        """
        device = next(self.parameters()).device
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1

        # Debug print: report number of time steps for this batch.
        print(f"[DEBUG] Processing batch of {batch_size} records with {n_steps} time steps each.")

        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        total_loss = 0.0
        total_tests = 0

        for t in range(n_steps):
            # Print progress every 10 steps or on the first step.
            if t == 0 or (t+1) % 10 == 0 or (t+1) == n_steps:
                print(f"[DEBUG] Time step {t+1}/{n_steps}")

            # Prepare input tensors.
            inputX = torch.zeros(batch_size, self.n_questions * 2, device=device)
            inputY = torch.zeros(batch_size, self.n_questions, device=device)
            truth = torch.zeros(batch_size, device=device)
            valid_mask = torch.zeros(batch_size, device=device)  # 1 if valid at time t

            for i, record in enumerate(batch):
                if t + 1 < record['n_answers']:
                    valid_mask[i] = 1
                    raw_q_current = record['questionId'][t]
                    raw_q_next = record['questionId'][t+1]
                    if self.question_mapping is not None:
                        q_current = self.question_mapping.get(raw_q_current, None)
                        q_next = self.question_mapping.get(raw_q_next, None)
                        if q_current is None or q_next is None:
                            continue
                    else:
                        q_current = raw_q_current
                        q_next = raw_q_next
                    correct_current = record['correct'][t]
                    correct_next = record['correct'][t+1]
                    # Compute index into inputX: (correct_current * n_questions) + (q_current - 1)
                    idx = int(correct_current * self.n_questions + (q_current - 1))
                    if idx < 0 or idx >= 2 * self.n_questions:
                        continue
                    inputX[i, idx] = 1
                    inputY[i, q_next - 1] = 1
                    truth[i] = correct_next

            if self.compressed_sensing:
                inputX = inputX @ self.basis

            pred, loss, hidden = self.cell(state, inputX, inputY, truth)
            state = hidden
            total_loss += loss
            total_tests += valid_mask.sum().item()

            # Debug: report valid tests at this time step.
            if (t+1) % 10 == 0 or (t+1) == n_steps:
                print(f"[DEBUG] Time step {t+1}: valid test count = {int(valid_mask.sum().item())}")

        return total_loss, total_tests

    def get_prediction_truth(self, batch):
        """
        Runs the model on a batch and collects prediction–truth pairs.
        Returns a list of dicts, each with keys 'pred' and 'truth'.
        """
        device = next(self.parameters()).device
        self.eval()
        predictions = []
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1

        # Debug print.
        print(f"[DEBUG] Evaluating {batch_size} records for {n_steps} time steps.")

        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        with torch.no_grad():
            for t in range(n_steps):
                # Optional: print progress every 10 steps.
                if t == 0 or (t+1) % 10 == 0 or (t+1) == n_steps:
                    print(f"[DEBUG] Evaluation time step {t+1}/{n_steps}")

                inputX = torch.zeros(batch_size, self.n_questions * 2, device=device)
                inputY = torch.zeros(batch_size, self.n_questions, device=device)
                truth_tensor = torch.zeros(batch_size, device=device)
                valid_mask = torch.zeros(batch_size, device=device)

                for i, record in enumerate(batch):
                    if t + 1 < record['n_answers']:
                        valid_mask[i] = 1
                        raw_q_current = record['questionId'][t]
                        raw_q_next = record['questionId'][t+1]
                        if self.question_mapping is not None:
                            q_current = self.question_mapping.get(raw_q_current, None)
                            q_next = self.question_mapping.get(raw_q_next, None)
                            if q_current is None or q_next is None:
                                continue
                        else:
                            q_current = raw_q_current
                            q_next = raw_q_next
                        correct_current = record['correct'][t]
                        correct_next = record['correct'][t+1]
                        idx = int(correct_current * self.n_questions + (q_current - 1))
                        if idx < 0 or idx >= 2 * self.n_questions:
                            continue
                        inputX[i, idx] = 1
                        inputY[i, q_next - 1] = 1
                        truth_tensor[i] = correct_next

                if self.compressed_sensing:
                    inputX = inputX @ self.basis

                pred, _, hidden = self.cell(state, inputX, inputY, truth_tensor)
                state = hidden
                # For each student in the batch, if valid, record prediction and truth.
                for i in range(batch_size):
                    if valid_mask[i] == 1:
                        predictions.append({'pred': pred[i].item(), 'truth': truth_tensor[i].item()})
        return predictions


def evaluate_model(model, test_batch):
    """
    Evaluates the model on test data.
    Returns:
      auc: Area under the ROC curve
      accuracy: classification accuracy (threshold 0.5)
    """
    preds_truth = model.get_prediction_truth(test_batch)
    if len(preds_truth) == 0:
        return 0.0, 0.0
    y_pred = [pt['pred'] for pt in preds_truth]
    y_true = [pt['truth'] for pt in preds_truth]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        auc = 0.0
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_true, y_pred_bin)
    return auc, accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DKTDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of student records, each being a dictionary.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# A collate function that just returns the batch as a list:
def collate_fn(batch):
    return batch


if __name__ == '__main__':
    # Get data.
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()

    # Define mini-batch size for training.
    mini_batch_size = 100

    # Create the model.
    model = DKT(n_questions=data.n_questions, n_hidden=200, dropout_pred=True,
                compressed_sensing=True, compressed_dim=100,
                question_mapping=data.question_mapping)

    device = get_device()
    model = model.to(device)

    # Use an optimizer for training.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 300

    # Training loop using mini-batches with debug prints and timing.
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tests = 0

        batches = semi_sorted_mini_batches(train_data, mini_batch_size, trim_to_batch_size=True)
        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()

            optimizer.zero_grad()
            loss, tests = model(batch)  # forward pass on the mini-batch
            loss.backward()             # backward pass on the mini-batch
            optimizer.step()            # update parameters

            total_loss += loss.item()   # accumulate scalar loss value
            total_tests += tests

            batch_end_time = time.time()
            print(f"[DEBUG] Epoch {epoch}, batch {batch_idx}/{len(batches)} processed in {batch_end_time - batch_start_time:.2f} sec, valid tests: {tests}")

        epoch_end_time = time.time()
        avg_loss = total_loss / total_tests if total_tests > 0 else total_loss
        print(f"[DEBUG] Epoch {epoch} took {epoch_end_time - epoch_start_time:.2f} sec, Avg Loss: {avg_loss:.4f}")

        # Evaluation.
        auc, acc = evaluate_model(model, test_data)
        print(f"Epoch {epoch}: auROC {auc:.4f}, Accuracy {acc:.4f}")
