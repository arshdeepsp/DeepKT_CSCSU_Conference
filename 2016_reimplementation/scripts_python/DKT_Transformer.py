import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

from data_assist import DataAssistMatrix

###############################################################################
# Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

###############################################################################
# Helper: Generate Causal Mask
###############################################################################
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)
    return mask

###############################################################################
# Optimized TransformerDKT Model
###############################################################################
class TransformerDKT(nn.Module):
    def __init__(self, n_questions, n_hidden, interaction_embed_dim=200,
                 n_layers=2, nhead=4, dropout=0.1, question_mapping=None):
        super(TransformerDKT, self).__init__()
        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.question_mapping = question_mapping

        # Reserve a special token for padding
        self.padding_idx = 2 * n_questions  
        self.interaction_embedding = nn.Embedding(2 * n_questions + 1, 
                                                  interaction_embed_dim,
                                                  padding_idx=self.padding_idx)
        if interaction_embed_dim != n_hidden:
            self.embedding_projection = nn.Linear(interaction_embed_dim, n_hidden)
        else:
            self.embedding_projection = None

        self.pos_encoder = PositionalEncoding(n_hidden, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.prediction_layer = nn.Linear(n_hidden, n_questions)
    
    def forward(self, batch):
        """
        batch: dict with keys:
          'input_seqs': LongTensor (batch, max_seq_len)
          'target_questions': LongTensor (batch, max_seq_len)
          'target_correctness': FloatTensor (batch, max_seq_len)
          'valid_mask': BoolTensor (batch, max_seq_len)
        """
        device = next(self.parameters()).device
        input_seqs = batch['input_seqs'].to(device)
        target_questions = batch['target_questions'].to(device)
        target_correctness = batch['target_correctness'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        batch_size, max_len = input_seqs.shape

        # Get embeddings and project if needed.
        emb = self.interaction_embedding(input_seqs)  # (batch, max_len, embed_dim)
        if self.embedding_projection is not None:
            emb = self.embedding_projection(emb)
        # Transformer expects (max_len, batch, n_hidden)
        emb = emb.transpose(0, 1)
        emb = self.pos_encoder(emb)

        # Create causal mask (same for all samples in batch).
        causal_mask = generate_square_subsequent_mask(max_len, device)
        # key_padding_mask: True for padded positions.
        key_padding_mask = ~valid_mask  # shape: (batch, max_len)

        transformer_output = self.transformer_encoder(emb, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        transformer_output = transformer_output.transpose(0, 1)  # (batch, max_len, n_hidden)

        logits = self.prediction_layer(transformer_output)  # (batch, max_len, n_questions)
        preds = torch.sigmoid(logits)
        # For each time step, select the prediction corresponding to the target question.
        preds_target = preds.gather(dim=2, index=target_questions.unsqueeze(-1)).squeeze(-1)  # (batch, max_len)

        loss = F.binary_cross_entropy(preds_target[valid_mask],
                                      target_correctness[valid_mask], reduction='sum')
        total_tests = valid_mask.sum().item()
        return loss, total_tests

    def get_prediction_truth(self, batch):
        # Similar to forward, but returns a list of prediction–truth pairs.
        device = next(self.parameters()).device
        self.eval()
        predictions = []
        with torch.no_grad():
            input_seqs = batch['input_seqs'].to(device)
            target_questions = batch['target_questions'].to(device)
            target_correctness = batch['target_correctness'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            batch_size, max_len = input_seqs.shape
            emb = self.interaction_embedding(input_seqs)
            if self.embedding_projection is not None:
                emb = self.embedding_projection(emb)
            emb = emb.transpose(0, 1)
            emb = self.pos_encoder(emb)
            causal_mask = generate_square_subsequent_mask(max_len, device)
            key_padding_mask = ~valid_mask
            transformer_output = self.transformer_encoder(emb, mask=causal_mask, src_key_padding_mask=key_padding_mask)
            transformer_output = transformer_output.transpose(0, 1)
            logits = self.prediction_layer(transformer_output)
            preds = torch.sigmoid(logits)
            preds_target = preds.gather(dim=2, index=target_questions.unsqueeze(-1)).squeeze(-1)

            for i in range(batch_size):
                seq_len = valid_mask[i].sum().item()
                for t in range(seq_len):
                    predictions.append({'pred': preds_target[i, t].item(),
                                        'truth': target_correctness[i, t].item()})
        return predictions

###############################################################################
# Optimized Collate Function
###############################################################################
def collate_fn(batch, question_mapping=None, padding_idx=None):
    """
    Precompute padded sequences for the Transformer model.
    Each record in the batch is expected to have keys: 'n_answers', 'questionId', 'correct'
    """
    input_seqs, target_questions, target_correctness = [], [], []
    for record in batch:
        n_answers = record['n_answers']
        if n_answers < 2:
            continue
        interactions = []
        tq = []
        tc = []
        for t in range(n_answers - 1):
            raw_q_current = record['questionId'][t]
            # Map question IDs if mapping provided.
            if question_mapping is not None:
                q_current = question_mapping.get(raw_q_current, None)
                if q_current is None:
                    continue
            else:
                q_current = raw_q_current
            idx = int(2 * (q_current - 1) + record['correct'][t])
            interactions.append(torch.tensor(idx, dtype=torch.long))
            
            raw_q_next = record['questionId'][t+1]
            if question_mapping is not None:
                q_next = question_mapping.get(raw_q_next, None)
                if q_next is None:
                    continue
            else:
                q_next = raw_q_next
            tq.append(torch.tensor(q_next - 1, dtype=torch.long))
            tc.append(torch.tensor(record['correct'][t+1], dtype=torch.float))
        if len(interactions) > 0:
            input_seqs.append(torch.stack(interactions))
            target_questions.append(torch.stack(tq))
            target_correctness.append(torch.stack(tc))
    if len(input_seqs) == 0:
        return None

    # Pad sequences.
    padded_input = pad_sequence(input_seqs, batch_first=True, padding_value=padding_idx)
    padded_tq = pad_sequence(target_questions, batch_first=True, padding_value=0)
    padded_tc = pad_sequence(target_correctness, batch_first=True, padding_value=-1)
    valid_mask = (padded_tc != -1)
    return {
        "input_seqs": padded_input,
        "target_questions": padded_tq,
        "target_correctness": padded_tc,
        "valid_mask": valid_mask
    }

###############################################################################
# Dataset
###############################################################################
class DKTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

###############################################################################
# Evaluation Function (unchanged)
###############################################################################
def evaluate_model(model, test_batch):
    preds_truth = model.get_prediction_truth(test_batch)
    if len(preds_truth) == 0:
        return 0.0, 0.0
    y_pred = [pt['pred'] for pt in preds_truth]
    y_true = [pt['truth'] for pt in preds_truth]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0.0
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_true, y_pred_bin)
    return auc, accuracy

###############################################################################
# Training Loop
###############################################################################
if __name__ == '__main__':
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()

    batch_size = 32
    train_dataset = DKTDataset(train_data)
    # Pass the mapping and padding_idx to collate_fn via a lambda.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda batch: collate_fn(batch, data.question_mapping, 2 * data.n_questions))
    
    model = TransformerDKT(n_questions=data.n_questions, n_hidden=200, interaction_embed_dim=200,
                             n_layers=2, nhead=4, dropout=0.1, question_mapping=data.question_mapping)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    num_epochs = 300

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tests = 0
        batch_idx = 0
        for raw_batch in train_loader:
            if raw_batch is None:
                continue
            optimizer.zero_grad()
            loss, tests = model(raw_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_tests += tests
            batch_idx += 1
            batch_loss = loss.item() / tests if tests > 0 else loss.item()
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {batch_loss:.4f}")
        avg_loss = epoch_loss / epoch_tests if epoch_tests > 0 else epoch_loss
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        scheduler.step(avg_loss)

        # Prepare test batch similarly.
        test_batch = collate_fn(test_data, data.question_mapping, 2 * data.n_questions)
        auc, acc = evaluate_model(model, test_batch)
        print(f"Epoch {epoch}: auROC = {auc:.4f}, Accuracy = {acc:.4f}")
