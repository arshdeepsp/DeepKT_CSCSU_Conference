import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

from data_assist import DataAssistMatrix

###############################################################################
# Improved DKTCell (same as before)
###############################################################################
class ImprovedDKTCell(nn.Module):
    def __init__(self, n_input, n_hidden, n_questions, dropout_pred=True):
        super(ImprovedDKTCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_questions = n_questions

        # Separate linear transformations for state and input
        self.state_transform = nn.Linear(n_hidden, n_hidden)
        self.input_transform = nn.Linear(n_input, n_hidden)
        
        # Additional hidden layers for increased model capacity
        self.hidden_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Prediction head with optional dropout
        pred_layers = []
        if dropout_pred:
            pred_layers.append(nn.Dropout(0.5))
        pred_layers.append(nn.Linear(n_hidden, n_questions))
        self.prediction_layer = nn.Sequential(*pred_layers)

    def forward(self, state, inputX, inputY, truth):
        state_feat = self.state_transform(state)
        input_feat = self.input_transform(inputX)
        combined = torch.tanh(state_feat + input_feat)
        hidden = self.hidden_layer(combined)
        logits = self.prediction_layer(hidden)
        pred_output = torch.sigmoid(logits)
        pred = torch.sum(pred_output * inputY, dim=1)
        loss = F.binary_cross_entropy(pred, truth, reduction='sum')
        return pred, loss, hidden

###############################################################################
# Improved DKT (same as before, with minor modifications for learning rate adjustments)
###############################################################################
class ImprovedDKT(nn.Module):
    def __init__(self, n_questions, n_hidden, interaction_embed_dim=200,
                 dropout_pred=True, question_mapping=None):
        super(ImprovedDKT, self).__init__()
        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.question_mapping = question_mapping

        self.input_dim = interaction_embed_dim
        self.interaction_embedding = nn.Embedding(2 * n_questions, self.input_dim)
        self.embedding_dropout = nn.Dropout(0.5)
        self.cell = ImprovedDKTCell(self.input_dim, n_hidden, n_questions, dropout_pred)
        self.start_layer = nn.Linear(1, n_hidden)

    def forward(self, batch):
        device = next(self.parameters()).device
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1

        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        total_loss = 0.0
        total_tests = 0

        for t in range(n_steps):
            inputX_int = torch.zeros(batch_size, device=device, dtype=torch.long)
            inputY = torch.zeros(batch_size, self.n_questions, device=device)
            truth = torch.zeros(batch_size, device=device)
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
                    idx = int(2 * (q_current - 1) + correct_current)
                    if idx < 0 or idx >= 2 * self.n_questions:
                        continue
                    inputX_int[i] = idx
                    inputY[i, q_next - 1] = 1
                    truth[i] = correct_next

            x = self.interaction_embedding(inputX_int)
            x = self.embedding_dropout(x)
            pred, loss, hidden = self.cell(state, x, inputY, truth)
            state = hidden
            total_loss += loss
            total_tests += valid_mask.sum().item()

        return total_loss, total_tests

    def get_prediction_truth(self, batch):
        device = next(self.parameters()).device
        self.eval()
        predictions = []
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1

        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        for t in range(n_steps):
            inputX_int = torch.zeros(batch_size, device=device, dtype=torch.long)
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
                    idx = int(2 * (q_current - 1) + correct_current)
                    if idx < 0 or idx >= 2 * self.n_questions:
                        continue
                    inputX_int[i] = idx
                    inputY[i, q_next - 1] = 1
                    truth_tensor[i] = correct_next

            x = self.interaction_embedding(inputX_int)
            x = self.embedding_dropout(x)
            pred, _, hidden = self.cell(state, x, inputY, truth_tensor)
            state = hidden
            for i in range(batch_size):
                if valid_mask[i] == 1:
                    predictions.append({'pred': pred[i].item(), 'truth': truth_tensor[i].item()})
        return predictions

###############################################################################
# Evaluation function
###############################################################################
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

###############################################################################
# Utility: Dataset and Collate Function
###############################################################################
class DKTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

###############################################################################
# Training Loop with Lowered Learning Rate and Scheduler
###############################################################################
if __name__ == '__main__':
    # Load data.
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()

    batch_size = 32
    train_dataset = DKTDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create the improved DKT model.
    model = ImprovedDKT(n_questions=data.n_questions, n_hidden=200, interaction_embed_dim=200,
                        dropout_pred=True, question_mapping=data.question_mapping)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Lower the learning rate from 0.01 to 0.001.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Optionally, add a scheduler to reduce the LR if the training loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    num_epochs = 300

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tests = 0
        batch_idx = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, tests = model(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_tests += tests
            batch_idx += 1
            batch_loss = loss.item() / tests if tests > 0 else loss.item()
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {batch_loss:.4f}")
        avg_loss = epoch_loss / epoch_tests if epoch_tests > 0 else epoch_loss
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        
        # Update the learning rate scheduler based on the average loss.
        scheduler.step(avg_loss)

        # Evaluate on test data after each epoch.
        auc, acc = evaluate_model(model, test_data)
        print(f"Epoch {epoch}: auROC = {auc:.4f}, Accuracy = {acc:.4f}")
