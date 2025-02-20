import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from data_assist import DataAssistMatrix
from util_exp import semi_sorted_mini_batches
from torch.utils.data import Dataset, DataLoader

#########################################
# Positional Encoding (sine/cosine style)
#########################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on 
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, last column remains zero; alternatively you could trim
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#########################################################
# Transformer-based Deep Knowledge Tracing Model
#########################################################
class DKTTransformer(nn.Module):
    """
    Transformer version of the Deep Knowledge Tracing (DKT) model.
    Instead of processing one time step at a time with recurrence,
    this model encodes an entire sequence (padded to the same length)
    using a Transformer with 4 blocks and 4 attention heads.
    
    For each time step t, the input vector is constructed from the student’s
    response at time t (a one-hot vector over 2*n_questions, or a compressed
    version thereof) and the model predicts the correctness at time t+1.
    
    Args:
      - n_questions: total number of distinct questions.
      - n_hidden: dimension of the hidden state (also d_model for Transformer).
      - dropout_pred: dropout rate (used both in the Transformer and in output).
      - compressed_sensing (bool): if True, use a compressed projection for the input.
      - compressed_dim (int): projected input dimension if compressed_sensing is True.
      - question_mapping (dict): mapping from raw question IDs to contiguous 1-indexed IDs.
    """
    def __init__(self, n_questions, n_hidden, dropout_pred=False,
                 compressed_sensing=False, compressed_dim=None,
                 question_mapping=None):
        super(DKTTransformer, self).__init__()
        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.dropout_pred = dropout_pred
        self.compressed_sensing = compressed_sensing
        self.question_mapping = question_mapping

        # Determine input dimension.
        if self.compressed_sensing:
            assert compressed_dim is not None, "compressed_dim must be specified"
            self.n_input = compressed_dim
            torch.manual_seed(12345)
            # Random projection matrix (non-trainable)
            self.basis = nn.Parameter(torch.randn(n_questions * 2, self.n_input), requires_grad=False)
        else:
            self.n_input = n_questions * 2
            self.basis = None

        # Project the (possibly compressed) input to d_model (n_hidden)
        self.input_proj = nn.Linear(self.n_input, n_hidden)
        # Positional encoding (using a fixed maximum length)
        self.positional_encoding = PositionalEncoding(d_model=n_hidden, dropout=0.1, max_len=500)
        # Transformer encoder with 4 blocks and 4 heads.
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, dropout=dropout_pred, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # Output layer projects the Transformer output to predictions for each question.
        self.output_layer = nn.Linear(n_hidden, n_questions)

    def forward(self, batch):
        """
        Forward pass on a batch of student records.
        For each record, we build a sequence of length (n_answers - 1) where:
           - The input at time t is a one-hot (or compressed) vector indicating the student's response at t.
           - The target (via inputY) is a one-hot vector for the next question.
           - The truth is the correctness for the next question.
        We pad sequences to the same length.
        
        Returns:
          total_loss: scalar loss (sum over valid prediction points).
          total_tests: total number of prediction points.
        """
        device = next(self.parameters()).device
        batch_size = len(batch)
        max_steps = max(record['n_answers'] for record in batch) - 1

        print(f"[DEBUG] Processing batch of {batch_size} records with {max_steps} time steps each.")

        # Initialize padded tensors.
        # inputX: for responses at time t, shape (batch_size, max_steps, n_questions*2)
        inputX_tensor = torch.zeros(batch_size, max_steps, self.n_questions * 2, device=device)
        # inputY: one-hot for next question at time t+1, shape (batch_size, max_steps, n_questions)
        inputY_tensor = torch.zeros(batch_size, max_steps, self.n_questions, device=device)
        # truth: binary correctness for next question.
        truth_tensor = torch.zeros(batch_size, max_steps, device=device)
        # valid_mask: marks valid (non-padded) time steps.
        valid_mask = torch.zeros(batch_size, max_steps, device=device)

        for i, record in enumerate(batch):
            n_steps = record['n_answers'] - 1
            for t in range(n_steps):
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
                # Compute index into input vector.
                idx = int(correct_current * self.n_questions + (q_current - 1))
                if idx < 0 or idx >= 2 * self.n_questions:
                    continue
                inputX_tensor[i, t, idx] = 1
                inputY_tensor[i, t, q_next - 1] = 1
                truth_tensor[i, t] = correct_next
                valid_mask[i, t] = 1

        # Apply compressed sensing projection if enabled.
        if self.compressed_sensing:
            inputX_tensor = inputX_tensor @ self.basis  # (batch, max_steps, n_input)

        # Project input to d_model.
        x = self.input_proj(inputX_tensor)  # (batch, max_steps, n_hidden)
        # Add positional encoding.
        x = self.positional_encoding(x)

        # Create causal (subsequent) mask so that each time step can attend only to previous ones.
        seq_len = max_steps
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

        # Pass the entire sequence through the Transformer.
        transformer_out = self.transformer(x, mask=causal_mask)  # (batch, max_steps, n_hidden)
        # Compute predictions via the output layer.
        logits = self.output_layer(transformer_out)  # (batch, max_steps, n_questions)
        pred_output = torch.sigmoid(logits)
        # For each time step, sum over the question dimension weighted by inputY_tensor to get a scalar prediction.
        pred = torch.sum(pred_output * inputY_tensor, dim=-1)  # (batch, max_steps)

        # Compute loss only over valid time steps.
        valid_pred = pred[valid_mask == 1]
        valid_truth = truth_tensor[valid_mask == 1]
        if valid_pred.numel() > 0:
            loss = F.binary_cross_entropy(valid_pred, valid_truth, reduction='sum')
        else:
            loss = torch.tensor(0.0, device=device)
        total_tests = valid_mask.sum().item()

        return loss, total_tests

    def get_prediction_truth(self, batch):
        """
        Runs the model in evaluation mode on a batch and collects prediction–truth pairs.
        Returns a list of dicts, each with keys 'pred' and 'truth'.
        """
        device = next(self.parameters()).device
        self.eval()
        batch_size = len(batch)
        max_steps = max(record['n_answers'] for record in batch) - 1

        print(f"[DEBUG] Evaluating {batch_size} records for {max_steps} time steps.")

        inputX_tensor = torch.zeros(batch_size, max_steps, self.n_questions * 2, device=device)
        inputY_tensor = torch.zeros(batch_size, max_steps, self.n_questions, device=device)
        truth_tensor = torch.zeros(batch_size, max_steps, device=device)
        valid_mask = torch.zeros(batch_size, max_steps, device=device)

        for i, record in enumerate(batch):
            n_steps = record['n_answers'] - 1
            for t in range(n_steps):
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
                inputX_tensor[i, t, idx] = 1
                inputY_tensor[i, t, q_next - 1] = 1
                truth_tensor[i, t] = correct_next
                valid_mask[i, t] = 1

        if self.compressed_sensing:
            inputX_tensor = inputX_tensor @ self.basis

        x = self.input_proj(inputX_tensor)
        x = self.positional_encoding(x)
        seq_len = max_steps
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        transformer_out = self.transformer(x, mask=causal_mask)
        logits = self.output_layer(transformer_out)
        pred_output = torch.sigmoid(logits)
        pred = torch.sum(pred_output * inputY_tensor, dim=-1)  # (batch, max_steps)

        predictions = []
        for i in range(batch_size):
            for t in range(max_steps):
                if valid_mask[i, t] == 1:
                    predictions.append({'pred': pred[i, t].item(), 'truth': truth_tensor[i, t].item()})
        return predictions

#########################################
# Evaluation and Helper Functions
#########################################
def evaluate_model(model, test_batch):
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

#########################################
# Dataset and Collate Function
#########################################
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

def collate_fn(batch):
    return batch

#########################################
# Main Training Loop
#########################################
if __name__ == '__main__':
    # Load data.
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()

    # Define mini-batch size.
    mini_batch_size = 100

    # Create the Transformer-based DKT model.
    model = DKTTransformer(n_questions=data.n_questions, n_hidden=200, dropout_pred=True,
                           compressed_sensing=True, compressed_dim=100,
                           question_mapping=data.question_mapping)

    device = get_device()
    model = model.to(device)

    # Use an optimizer.
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

        # Evaluation on test data.
        auc, acc = evaluate_model(model, test_data)
        print(f"Epoch {epoch}: auROC {auc:.4f}, Accuracy {acc:.4f}")
