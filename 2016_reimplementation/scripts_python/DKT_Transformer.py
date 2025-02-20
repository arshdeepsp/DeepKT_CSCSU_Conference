import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, key_padding_mask=None):
        # x: (batch, seq_len, embed_dim) -> transpose for MultiheadAttention: (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        # Use both the causal mask and key_padding_mask.
        attn_output, _ = self.attention(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        # Transpose back to (batch, seq_len, embed_dim)
        return out2.transpose(0, 1)

class DKT(nn.Module):
    def __init__(self, n_questions, embed_dim=128, num_heads=8, ff_dim=256, 
                 num_transformer_blocks=4, dropout_rate=0.1, max_seq_len=1000,
                 question_mapping=None):
        super(DKT, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.n_questions = n_questions
        self.embed_dim = embed_dim
        self.question_mapping = question_mapping

        # Reserve an extra index for padding in questions.
        self.question_embedding = nn.Embedding(n_questions + 1, embed_dim // 2, padding_idx=n_questions)
        # For correctness, valid values are 0 and 1; use index 2 as padding.
        self.correct_embedding = nn.Embedding(3, embed_dim // 2, padding_idx=2)
        
        # Pre-compute positional encodings.
        self.register_buffer("pos_encoding", self._positional_encoding(max_seq_len, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ])
        
        self.output_layer = nn.Linear(embed_dim, n_questions)
        self.dropout = nn.Dropout(dropout_rate)

    def _positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def _create_causal_mask(self, size, device):
        # Create a boolean mask: True for positions that should be masked (future tokens).
        # Here, positions (i,j) with j > i are masked.
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def _process_sequence(self, questions, correctness, key_padding_mask=None):
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Get embeddings and concatenate.
        q_embeds = self.question_embedding(questions)
        c_embeds = self.correct_embedding(correctness)
        x = torch.cat([q_embeds, c_embeds], dim=-1)
        
        # Add positional encoding.
        x = x + self.pos_encoding[:seq_len].to(device)
        
        # Create a causal mask.
        attention_mask = self._create_causal_mask(seq_len, device)
        
        # Process through transformer blocks.
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=attention_mask, key_padding_mask=key_padding_mask)
        
        x = self.dropout(x)
        return self.output_layer(x)

    def forward(self, batch):
        device = next(self.parameters()).device
        batch_size = len(batch)
        n_steps = max(record['n_answers'] for record in batch) - 1
        
        if n_steps < 1:
            return torch.tensor(0.0, device=device), 0

        # Initialize tensors.
        # For questions, use padding index (n_questions); for correctness, padding index 2.
        questions = torch.full((batch_size, n_steps), self.n_questions, dtype=torch.long, device=device)
        correctness = torch.full((batch_size, n_steps), 2, dtype=torch.long, device=device)
        # For next_questions, initialize with 0 (a valid dummy index) so that padded positions won't cause indexing errors.
        next_questions = torch.zeros((batch_size, n_steps), dtype=torch.long, device=device)
        truth = torch.zeros(batch_size, n_steps, device=device)
        valid_mask = torch.zeros(batch_size, n_steps, device=device)  # 1 indicates a valid timestep.

        # Fill tensors with data from each record.
        for i, record in enumerate(batch):
            valid_count = 0
            for t in range(record['n_answers'] - 1):
                try:
                    q_current = record['questionId'][t]
                    q_next = record['questionId'][t+1]
                    
                    if self.question_mapping is not None:
                        q_current = self.question_mapping.get(q_current)
                        q_next = self.question_mapping.get(q_next)
                        if q_current is None or q_next is None:
                            continue
                    
                    q_current = int(q_current) - 1
                    q_next = int(q_next) - 1
                    
                    if not (0 <= q_current < self.n_questions and 0 <= q_next < self.n_questions):
                        continue
                    
                    correct = int(record['correct'][t])
                    next_correct = int(record['correct'][t+1])
                    
                    if valid_count >= n_steps:
                        break
                    
                    valid_mask[i, valid_count] = 1
                    questions[i, valid_count] = q_current
                    correctness[i, valid_count] = correct
                    next_questions[i, valid_count] = q_next
                    truth[i, valid_count] = next_correct
                    valid_count += 1
                
                except Exception as e:
                    continue

        # Build key_padding_mask for transformer: True where positions are padded.
        key_padding_mask = (valid_mask == 0).bool()
        
        # Get predictions (logits).
        logits = self._process_sequence(questions, correctness, key_padding_mask=key_padding_mask)
        # Gather predictions for the "next" question.
        pred_logits = torch.gather(logits, 2, next_questions.unsqueeze(-1)).squeeze(-1)
        
        # Compute loss using BCEWithLogitsLoss on valid positions.
        loss = F.binary_cross_entropy_with_logits(pred_logits, truth, reduction='none')
        loss = (loss * valid_mask).sum()
        total_tests = valid_mask.sum().item()
        
        return loss, total_tests

    def predict(self, questions, correctness, key_padding_mask=None):
        """Make predictions for a single sequence."""
        self.eval()
        with torch.no_grad():
            logits = self._process_sequence(questions, correctness, key_padding_mask=key_padding_mask)
            return torch.sigmoid(logits)

def evaluate_model(model, test_batch):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        all_preds = []
        all_truths = []
        
        # Process each sequence individually.
        for record in test_batch:
            seq_len = record['n_answers'] - 1
            if seq_len < 1:
                continue
            
            questions = torch.full((1, seq_len), model.n_questions, dtype=torch.long, device=device)
            correctness = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
            
            valid_pos = []
            for t in range(seq_len):
                try:
                    q_current = record['questionId'][t]
                    q_next = record['questionId'][t+1]
                    
                    if model.question_mapping is not None:
                        q_current = model.question_mapping.get(q_current)
                        q_next = model.question_mapping.get(q_next)
                        if q_current is None or q_next is None:
                            continue
                    
                    q_current = int(q_current) - 1
                    q_next = int(q_next) - 1
                    
                    if not (0 <= q_current < model.n_questions and 0 <= q_next < model.n_questions):
                        continue
                    
                    correct = int(record['correct'][t])
                    questions[0, t] = q_current
                    correctness[0, t] = correct
                    valid_pos.append((t, q_next, record['correct'][t+1]))
                
                except Exception as e:
                    continue
            
            if not valid_pos:
                continue
            
            # For evaluation, assume full validity so no key_padding_mask.
            pred_output = model.predict(questions, correctness)
            
            for t, q_next, truth in valid_pos:
                pred = pred_output[0, t, q_next].item()
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

class DKTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

if __name__ == '__main__':
    from data_assist import DataAssistMatrix
    
    # Load data.
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()
    
    # Updated model parameters.
    n_questions = data.n_questions
    batch_size = 32
    num_epochs = 100
    embed_dim = 128              # Increased embedding dimension.
    num_heads = 8                # embed_dim must be divisible by num_heads.
    ff_dim = 256                 # Feed-forward network dimension.
    num_transformer_blocks = 4   # Reduced transformer blocks.
    dropout_rate = 0.1           # Lower dropout.
    learning_rate = 0.0005       # Lower learning rate.
    max_seq_len = 200
    
    print(f"\nModel configuration:")
    print(f"Number of questions: {n_questions}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Feed-forward dimension: {ff_dim}")
    print(f"Transformer blocks: {num_transformer_blocks}")
    print(f"Max sequence length: {max_seq_len}")
    
    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Truncate sequences.
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
    
    # Initialize model.
    model = DKT(
        n_questions=n_questions,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len,
        question_mapping=data.question_mapping
    ).to(device)
    
    # Create DataLoader.
    train_dataset = DKTDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add a learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    
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
            
            # Evaluate after each epoch.
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
            
            # Update learning rate based on epoch loss.
            scheduler.step(epoch_loss)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        if best_model is not None:
            model.load_state_dict(best_model)
            print(f"\nTraining completed. Best AUC: {best_auc:.4f}")
            
            # Final evaluation.
            model.eval()
            with torch.no_grad():
                final_auc, final_accuracy = evaluate_model(model, test_data)
                print(f"Final Test AUC: {final_auc:.4f}, Accuracy: {final_accuracy:.4f}")
