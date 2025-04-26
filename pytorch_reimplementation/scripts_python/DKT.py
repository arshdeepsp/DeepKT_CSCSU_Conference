import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import time
import datetime

from data_assist import DataAssistMatrix

###############################################################################
# Utility Functions for Timing and Model Info
###############################################################################
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    """Format time in seconds to a readable string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

###############################################################################
# Improved DKTCell
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
        """
        Args:
          state: Tensor of shape (batch, n_hidden)
          inputX: Tensor of shape (batch, n_input) – the embedded interaction
          inputY: Tensor of shape (batch, n_questions) – one-hot for the next question.
          truth:  Tensor of shape (batch,) – binary correctness (target)
        Returns:
          pred: Tensor (batch,) – predicted probability (scalar per sample)
          loss: scalar loss (sum over batch)
          hidden: new hidden state (batch, n_hidden)
        """
        # Process state and input separately
        state_feat = self.state_transform(state)
        input_feat = self.input_transform(inputX)
        
        # Combine features
        combined = torch.tanh(state_feat + input_feat)
        
        # Apply additional processing through hidden layers
        hidden = self.hidden_layer(combined)
        
        # Generate predictions
        logits = self.prediction_layer(hidden)
        pred_output = torch.sigmoid(logits)
        
        # Debug info
        if torch.isnan(pred_output).any() or torch.isinf(pred_output).any():
            print("Warning: NaN or Inf detected in sigmoid output")
            pred_output = torch.nan_to_num(pred_output, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Find the next question for each student in the batch
        # This assumes inputY is a one-hot encoding with exactly one 1 per row
        batch_indices = torch.arange(inputY.size(0), device=inputY.device)
        
        # Check if we have valid next questions
        row_sums = inputY.sum(dim=1)
        valid_rows = row_sums > 0  # Identify rows with at least one question marked
        
        if not valid_rows.all():
            # Handle the case where some rows don't have any question marked
            print(f"Warning: {(~valid_rows).sum().item()} rows don't have a next question")
        
        # For rows with no question, set a default (first question)
        next_q_indices = torch.zeros(inputY.size(0), dtype=torch.long, device=inputY.device)
        # Only use argmax for valid rows
        next_q_indices[valid_rows] = torch.argmax(inputY[valid_rows], dim=1)
        
        # Select the prediction for the next question only
        pred = pred_output[batch_indices, next_q_indices]
        
        # Extra safety: explicitly ensure predictions are in [0,1]
        pred = torch.clamp(pred, min=0.0, max=1.0)
        
        # Ensure truth is also in [0,1] range
        truth_clamped = torch.clamp(truth, min=0.0, max=1.0)
        
        # Use try-except to catch any remaining issues
        try:
            loss = F.binary_cross_entropy(pred, truth_clamped, reduction='sum')
        except RuntimeError as e:
            print(f"BCE Error: {e}")
            print(f"pred min: {pred.min().item()}, max: {pred.max().item()}")
            print(f"truth min: {truth_clamped.min().item()}, max: {truth_clamped.max().item()}")
            # Fallback to a safe loss calculation
            loss = torch.tensor(0.0, device=pred.device)
            
        return pred, loss, hidden

###############################################################################
# Improved DKT
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
        """
        Args:
          batch: list of student records (each a dict with keys:
                 'n_answers', 'questionId', 'correct')
        Returns:
          total_loss: scalar loss (sum over time and students)
          total_tests: total number of prediction points (for averaging)
        """
        device = next(self.parameters()).device
        batch_size = len(batch)
        
        # Safely compute max steps - ensure we have at least some records with multiple answers
        max_answers = max((record['n_answers'] for record in batch), default=0)
        if max_answers <= 1:
            # Not enough data for sequence prediction
            return torch.tensor(0.0, device=device), 0
            
        n_steps = max_answers - 1

        # Initialize the hidden state.
        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        total_loss = torch.tensor(0.0, device=device)
        total_tests = 0

        # Loop over time steps.
        for t in range(n_steps):
            # Prepare tensors for this time step.
            inputX_int = torch.zeros(batch_size, device=device, dtype=torch.long)
            inputY = torch.zeros(batch_size, self.n_questions, device=device)
            truth = torch.zeros(batch_size, device=device)
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Collect data for this time step
            for i, record in enumerate(batch):
                if t + 1 < record['n_answers']:
                    # Get current and next question IDs
                    try:
                        raw_q_current = record['questionId'][t]
                        raw_q_next = record['questionId'][t+1]
                        correct_current = int(record['correct'][t])
                        correct_next = float(record['correct'][t+1])
                        
                        # Apply question mapping if available
                        if self.question_mapping is not None:
                            q_current = self.question_mapping.get(raw_q_current, None)
                            q_next = self.question_mapping.get(raw_q_next, None)
                            if q_current is None or q_next is None:
                                continue
                        else:
                            q_current = raw_q_current
                            q_next = raw_q_next
                        
                        # Validate q_current and q_next are within valid range (1 to n_questions)
                        if not (1 <= q_current <= self.n_questions and 1 <= q_next <= self.n_questions):
                            continue
                            
                        # Validate correctness values
                        if not (0 <= correct_current <= 1 and 0 <= correct_next <= 1):
                            continue
                            
                        # Compute interaction index
                        idx = int(2 * (q_current - 1) + correct_current)
                        if 0 <= idx < 2 * self.n_questions:
                            inputX_int[i] = idx
                            inputY[i, q_next - 1] = 1
                            truth[i] = correct_next
                            valid_mask[i] = True
                    except (IndexError, ValueError, TypeError) as e:
                        # Skip problematic records
                        continue

            # Skip this step if there are no valid samples
            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue

            # Embed the integer indices into dense vectors
            x = self.interaction_embedding(inputX_int)  # shape: (batch, input_dim)
            x = self.embedding_dropout(x)
            
            # Process only valid samples
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            if valid_indices.numel() > 0:
                try:
                    x_valid = x[valid_indices]
                    inputY_valid = inputY[valid_indices]
                    truth_valid = truth[valid_indices]
                    state_valid = state[valid_indices]
                    
                    # Process through the cell
                    pred, loss, hidden = self.cell(state_valid, x_valid, inputY_valid, truth_valid)
                    
                    # Update states for valid indices
                    for i, idx in enumerate(valid_indices):
                        state[idx] = hidden[i]
                    
                    # Update total loss and test count
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss
                        total_tests += valid_count
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    # Continue with next time step on error
                    continue

        # Return 0 loss if no tests were processed
        if total_tests == 0:
            return torch.tensor(0.0, device=device), 0
            
        return total_loss, total_tests

    def get_prediction_truth(self, batch):
        """
        Runs the model on a batch and collects prediction–truth pairs.
        Returns:
          List of dicts with keys 'pred' and 'truth'
        """
        device = next(self.parameters()).device
        self.eval()
        predictions = []
        batch_size = len(batch)
        
        # Calculate max steps safely
        max_answers = max((record['n_answers'] for record in batch), default=0)
        if max_answers <= 1:
            return predictions
            
        n_steps = max_answers - 1

        # Memory usage tracking
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

        state = self.start_layer(torch.zeros(batch_size, 1, device=device))
        
        prediction_start_time = time.time()
        
        for t in range(n_steps):
            inputX_int = torch.zeros(batch_size, device=device, dtype=torch.long)
            inputY = torch.zeros(batch_size, self.n_questions, device=device)
            truth_tensor = torch.zeros(batch_size, device=device)
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for i, record in enumerate(batch):
                if t + 1 < record['n_answers']:
                    try:
                        raw_q_current = record['questionId'][t]
                        raw_q_next = record['questionId'][t+1]
                        correct_current = int(record['correct'][t])
                        correct_next = float(record['correct'][t+1])
                        
                        # Apply question mapping if available
                        if self.question_mapping is not None:
                            q_current = self.question_mapping.get(raw_q_current, None)
                            q_next = self.question_mapping.get(raw_q_next, None)
                            if q_current is None or q_next is None:
                                continue
                        else:
                            q_current = raw_q_current
                            q_next = raw_q_next
                        
                        # Validate q_current and q_next are within valid range
                        if not (1 <= q_current <= self.n_questions and 1 <= q_next <= self.n_questions):
                            continue
                            
                        # Validate correctness values
                        if not (0 <= correct_current <= 1 and 0 <= correct_next <= 1):
                            continue
                            
                        # Compute interaction index
                        idx = int(2 * (q_current - 1) + correct_current)
                        if 0 <= idx < 2 * self.n_questions:
                            inputX_int[i] = idx
                            inputY[i, q_next - 1] = 1
                            truth_tensor[i] = correct_next
                            valid_mask[i] = True
                    except (IndexError, ValueError, TypeError) as e:
                        # Skip problematic records
                        continue

            # Skip this step if there are no valid samples
            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue

            # Embed the vectors
            x = self.interaction_embedding(inputX_int)
            x = self.embedding_dropout(x)

            # Process only valid samples
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            if valid_indices.numel() > 0:
                try:
                    x_valid = x[valid_indices]
                    inputY_valid = inputY[valid_indices]
                    truth_valid = truth_tensor[valid_indices]
                    state_valid = state[valid_indices]
                    
                    # Process through the cell
                    pred, _, hidden = self.cell(state_valid, x_valid, inputY_valid, truth_valid)
                    
                    # Update states for valid indices
                    for i, idx in enumerate(valid_indices):
                        state[idx] = hidden[i]
                    
                    # Collect predictions
                    for j, idx in enumerate(valid_indices):
                        predictions.append({
                            'pred': pred[j].item(),
                            'truth': truth_valid[j].item()
                        })
                except Exception as e:
                    print(f"Error in evaluation step {t}: {e}")
                    continue
        
        prediction_time = time.time() - prediction_start_time
        
        # Memory usage
        end_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB
        
        if len(predictions) > 0:
            print(f"  Generated {len(predictions)} predictions in {prediction_time:.2f}s")
            if torch.cuda.is_available():
                print(f"  Memory used: {memory_used:.2f} MB")

        return predictions

###############################################################################
# Evaluation function
###############################################################################
def evaluate_model(model, test_batch):
    """
    Evaluate the model on test data, calculating AUC and accuracy metrics.
    
    Args:
        model: The DKT model
        test_batch: List of test records
        
    Returns:
        auc: Area under ROC curve
        accuracy: Classification accuracy
    """
    eval_start = time.time()
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for inference
        preds_truth = model.get_prediction_truth(test_batch)
    
    if len(preds_truth) == 0:
        return 0.0, 0.0
    
    y_pred = [pt['pred'] for pt in preds_truth]
    y_true = [pt['truth'] for pt in preds_truth]
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"Error in AUC calculation: {e}")
        auc = 0.0
    
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred]
    accuracy = accuracy_score(y_true, y_pred_bin)
    
    # Calculate number of correct predictions
    num_correct = sum(1 for true, pred in zip(y_true, y_pred_bin) if true == pred)
    total = len(y_true)
    
    eval_time = time.time() - eval_start
    print(f"  Evaluation: {total} predictions, {num_correct} correct, time: {eval_time:.2f}s")
    
    return auc, accuracy

###############################################################################
# Utility: Dataset and Collate Function
###############################################################################
class DKTDataset(Dataset):
    def __init__(self, data):
        """
        Args:
          data: list of student records (each a dict)
        """
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
    # Record start time
    start_time = time.time()
    
    # Print header
    print("="*50)
    print(f" ImprovedDKT Model Training - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # Load data.
    data = DataAssistMatrix()
    train_data = data.getTrainData()
    test_data = data.getTestData()
    
    print(f"total test questions: ", set(q for record in test_data for q in record['questionId']))
    print(f"total test questions number: ", len(test_data))
    print(f"total train questions number: ", len(train_data))

    # Create Dataset and DataLoader with a reduced batch size (e.g., 32).
    batch_size = 32
    train_dataset = DKTDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    data_loading_time = time.time() - start_time
    print(f"Data loading completed in {data_loading_time:.2f} seconds")

    # Create the improved DKT model.
    model = ImprovedDKT(n_questions=data.n_questions, n_hidden=200, interaction_embed_dim=200,
                        dropout_pred=True, question_mapping=data.question_mapping)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate and print model size
    num_params = count_parameters(model)
    print(f"Running on device: {device}")
    print(f"Model parameters: questions={data.n_questions}, hidden_dim=200")
    print(f"Total trainable parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Lower the learning rate from 0.01 to 0.001 for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 300
    
    # For early stopping
    best_auc = 0
    patience = 10
    patience_counter = 0
    
    # For learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize tracking variables
    total_training_time = 0
    epoch_times = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tests = 0
        batch_idx = 0
        
        # Start timer for this epoch
        epoch_start_time = time.time()
        
        for batch in train_loader:
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            try:
                loss, tests = model(batch)
                
                if tests > 0:  # Only backpropagate if we have valid predictions
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_tests += tests
                    batch_idx += 1
                    
                    # Calculate per-sample loss for display
                    batch_loss = loss.item() / tests if tests > 0 else 0
                    batch_time = time.time() - batch_start_time
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {batch_loss:.4f} (Time: {batch_time:.2f}s)")
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                # Skip problematic batches but continue training
                continue
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_training_time += epoch_time
        
        # Calculate epoch average loss
        avg_loss = epoch_loss / epoch_tests if epoch_tests > 0 else 0
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Time = {format_time(epoch_time)}")
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        # Evaluate on test data
        eval_start_time = time.time()
        try:
            auc, acc = evaluate_model(model, test_data)
            eval_time = time.time() - eval_start_time
            print(f"Epoch {epoch}: auROC = {auc:.4f}, Accuracy = {acc:.4f}, Eval Time = {eval_time:.2f}s")
            
            # Early stopping logic
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_improved_dkt_model.pt')
                print(f"New best model saved with auROC = {auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        except Exception as e:
            print(f"Evaluation error: {e}")
            # Continue training even if evaluation fails
            patience_counter += 1
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Average epoch time: {format_time(sum(epoch_times) / len(epoch_times))}")
    print(f"Fastest epoch: {format_time(min(epoch_times))}")
    print(f"Slowest epoch: {format_time(max(epoch_times))}")
    print(f"Model size: {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Load best model for final evaluation
    try:
        model.load_state_dict(torch.load('best_improved_dkt_model.pt'))
        final_auc, final_acc = evaluate_model(model, test_data)
        print(f"Final model performance: auROC = {final_auc:.4f}, Accuracy = {final_acc:.4f}")
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using last model for final evaluation")
        final_auc, final_acc = evaluate_model(model, test_data)
        print(f"Final model performance: auROC = {final_auc:.4f}, Accuracy = {final_acc:.4f}")