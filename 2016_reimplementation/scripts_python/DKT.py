import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

class DKT(nn.Module):
    def __init__(self, n_questions, embed_dim=256, hidden_dim=256, 
                 num_layers=2, dropout_rate=0.1, question_mapping=None):
        super(DKT, self).__init__()
        
        self.n_questions = n_questions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.question_mapping = question_mapping

        # Create separate embeddings for questions and correctness
        self.question_embedding = nn.Embedding(n_questions, embed_dim // 2)
        self.correct_embedding = nn.Embedding(2, embed_dim // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, n_questions)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch):
        device = next(self.parameters()).device
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1
        
        if n_steps < 1:
            return torch.tensor(0.0, device=device), 0

        # Initialize tensors for questions and correctness
        questions = torch.zeros(batch_size, n_steps, dtype=torch.long, device=device)
        correctness = torch.zeros(batch_size, n_steps, dtype=torch.long, device=device)
        next_questions = torch.zeros(batch_size, n_steps, dtype=torch.long, device=device)
        truth = torch.zeros(batch_size, n_steps, device=device)
        valid_mask = torch.zeros(batch_size, n_steps, device=device)

        # Fill the tensors
        for i, record in enumerate(batch):
            for t in range(min(n_steps, record['n_answers'] - 1)):
                try:
                    # Get current and next questions
                    q_current = record['questionId'][t]
                    q_next = record['questionId'][t+1]
                    
                    # Apply mapping if exists
                    if self.question_mapping is not None:
                        q_current = self.question_mapping.get(q_current)
                        q_next = self.question_mapping.get(q_next)
                        if q_current is None or q_next is None:
                            continue
                    
                    # Convert to 0-based indices
                    q_current = int(q_current) - 1
                    q_next = int(q_next) - 1
                    
                    if not (0 <= q_current < self.n_questions and 0 <= q_next < self.n_questions):
                        continue
                    
                    # Get correctness
                    correct = int(record['correct'][t])
                    next_correct = int(record['correct'][t+1])
                    
                    # Set values
                    valid_mask[i, t] = 1
                    questions[i, t] = q_current
                    correctness[i, t] = correct
                    next_questions[i, t] = q_next
                    truth[i, t] = next_correct
                
                except Exception as e:
                    print(f"Error processing record at position {i}, step {t}: {e}")
                    continue

        # Get embeddings
        q_embeds = self.question_embedding(questions)
        c_embeds = self.correct_embedding(correctness)
        
        # Combine embeddings
        x = torch.cat([q_embeds, c_embeds], dim=-1)
        
        # Apply LSTM
        x = self.dropout(x)
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        
        # Compute predictions
        logits = self.output_layer(outputs)
        pred_output = torch.sigmoid(logits)
        
        # Gather predictions for next questions
        pred = torch.gather(pred_output, 2, next_questions.unsqueeze(-1)).squeeze(-1)
        
        # Compute loss
        loss = F.binary_cross_entropy(pred, truth, reduction='none')
        loss = (loss * valid_mask).sum()
        total_tests = valid_mask.sum().item()
        
        return loss, total_tests

class DKTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

def evaluate_model(model, test_batch):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        batch_size = len(test_batch)
        n_steps = max(record['n_answers'] for record in test_batch) - 1
        
        if n_steps < 1:
            return 0.0, 0.0

        all_preds = []
        all_truths = []
        
        # Process each batch item
        for i, record in enumerate(test_batch):
            seq_len = min(n_steps, record['n_answers'] - 1)
            if seq_len < 1:
                continue
                
            # Create tensors for this sequence
            questions = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            correctness = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            
            # Fill sequence data
            for t in range(seq_len):
                try:
                    q_current = record['questionId'][t]
                    if model.question_mapping is not None:
                        q_current = model.question_mapping.get(q_current)
                        if q_current is None:
                            continue
                    
                    q_current = int(q_current) - 1
                    if not (0 <= q_current < model.n_questions):
                        continue
                        
                    correct = int(record['correct'][t])
                    
                    questions[0, t] = q_current
                    correctness[0, t] = correct
                    
                except Exception as e:
                    print(f"Error in evaluation at position {i}, step {t}: {e}")
                    continue
            
            # Get embeddings
            q_embeds = model.question_embedding(questions)
            c_embeds = model.correct_embedding(correctness)
            x = torch.cat([q_embeds, c_embeds], dim=-1)
            
            # Process through LSTM
            outputs, _ = model.lstm(x)
            logits = model.output_layer(outputs)
            pred_output = torch.sigmoid(logits)
            
            # Collect predictions for next questions
            for t in range(seq_len):
                q_next = record['questionId'][t+1]
                if model.question_mapping is not None:
                    q_next = model.question_mapping.get(q_next)
                    if q_next is None:
                        continue
                        
                q_next = int(q_next) - 1
                if not (0 <= q_next < model.n_questions):
                    continue
                
                # Get prediction for next question
                pred = pred_output[0, t, q_next].item()
                truth = record['correct'][t+1]
                
                all_preds.append(pred)
                all_truths.append(truth)

        if not all_preds:
            return 0.0, 0.0

        try:
            auc = roc_auc_score(all_truths, all_preds)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc = 0.0

        y_pred_bin = [1 if p > 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_truths, y_pred_bin)
        
        print(f"Number of test predictions: {len(all_preds)}")
        print(f"Prediction range: {min(all_preds):.4f} - {max(all_preds):.4f}")
        
        return auc, accuracy

if __name__ == '__main__':
    from data_assist import DataAssistMatrix
    
    # Load data
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()
    
    # Model parameters
    n_questions = data.n_questions
    batch_size = 32
    num_epochs = 30
    embed_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout_rate = 0.1
    learning_rate = 0.001
    max_seq_len = 200
    
    print(f"\nModel configuration:")
    print(f"Number of questions: {n_questions}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of LSTM layers: {num_layers}")
    print(f"Max sequence length: {max_seq_len}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Truncate sequences
    def truncate_sequence(record):
        if record['n_answers'] > max_seq_len + 1:
            return {
                'n_answers': max_seq_len + 1,
                'questionId': record['questionId'][:max_seq_len + 1],
                'correct': record['correct'][:max_seq_len + 1]
            }
        return record

    train_data = [truncate_sequence(record) for record in train_data]
    test_data = [truncate_sequence(record) for record in test_data]
    
    # Initialize model
    model = DKT(
        n_questions=n_questions,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        question_mapping=data.question_mapping
    ).to(device)
    
    # Create DataLoader
    train_dataset = DKTDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nStarting training...")
    best_auc = 0
    best_model = None
    patience = 10
    patience_counter = 0
    
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0
            total_samples = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss, num_samples = model(batch)
                
                avg_loss = loss / num_samples if num_samples > 0 else loss
                avg_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_samples += num_samples
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                          f"Loss: {total_loss/total_samples:.4f}")
            
            epoch_loss = total_loss / total_samples if total_samples > 0 else total_loss
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            # Evaluate after each epoch
            model.eval()
            with torch.no_grad():
                auc, accuracy = evaluate_model(model, test_data)
                print(f"Epoch {epoch} - Test AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = model.state_dict()
                    patience_counter = 0
                    print(f"New best AUC: {best_auc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping after {epoch} epochs")
                        break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        if best_model is not None:
            print(f"Training completed. Best AUC: {best_auc:.4f}")
