import torch
import gc
import random
import math

def get_n_tests(batch):
    """
    For a given batch (list of student answer dicts), compute the total number 
    of tests available. For each time step from 1 to n_steps (n_steps is the 
    maximum number of answers among students minus one), we build a mask over 
    the students and sum all entries.
    """
    n_steps = get_n_steps(batch)
    n_students = len(batch)
    # Create a zeros tensor of shape (n_students, n_steps)
    m = torch.zeros((n_students, n_steps))
    # Note: Lua uses 1-indexing. Here, we iterate i from 1 to n_steps and assign 
    # results to column (i-1) so that the behavior is equivalent.
    for i in range(1, n_steps + 1):
        mask = get_mask(batch, i)
        m[:, i - 1] = mask
    # Return the sum of all entries as a Python number.
    return m.sum().item()

def get_total_tests(batches, max_steps=None):
    """
    Given a list of batches, returns the total number of tests.
    If max_steps is specified and positive, each batch's tests are capped at max_steps.
    """
    total = 0
    for batch in batches:
        n_tests = get_n_tests(batch)
        if max_steps is not None and max_steps > 0:
            n_tests = min(max_steps, n_tests)
        total += n_tests
    return total

def get_n_steps(batch):
    """
    Returns the maximum number of steps (i.e., answers) in the batch minus one.
    Each element in batch is assumed to be a dict with key 'n_answers'.
    """
    max_steps = 0
    for ans in batch:
        if ans['n_answers'] > max_steps:
            max_steps = ans['n_answers']
    return max_steps - 1

def get_mask(batch, k):
    """
    For a given batch and time step k (1-indexed), return a tensor mask (shape: [len(batch)])
    that contains 1 if the student has at least (k + 1) answers and 0 otherwise.
    """
    n = len(batch)
    mask = torch.zeros(n)
    for i, ans in enumerate(batch):
        if k + 1 <= ans['n_answers']:
            mask[i] = 1
    return mask

def evaluate(rnn, data):
    """
    Evaluates the given RNN model on test data. The function:
      - Creates mini-batches from the test data.
      - Gets predictions (each with a 'pred' score and a binary 'truth').
      - Computes total positives/negatives, accuracy, and an approximate AUC.
    
    Note: The RNN is assumed to have a method `getPredictionTruth(batch)`
    that returns a list of prediction dicts.
    """
    mini_batches = semi_sorted_mini_batches(data.getTestData(), 100, trim_to_batch_size=False)
    all_predictions = []
    total_positives = 0
    total_negatives = 0

    for batch in mini_batches:
        predictions = rnn.get_prediction_truth(batch)
        for prediction in predictions:
            if prediction['truth'] == 1:
                total_positives += 1
            else:
                total_negatives += 1
            all_predictions.append(prediction)
        gc.collect()

    # Sort predictions in descending order of the predicted score.
    all_predictions.sort(key=lambda x: x['pred'], reverse=True)

    true_positives = 0
    false_positives = 0
    correct = 0
    auc = 0.0
    last_fpr = None
    last_tpr = None

    for i, p in enumerate(all_predictions, start=1):
        if p['truth'] == 1:
            true_positives += 1
        else:
            false_positives += 1

        guess = 1 if p['pred'] > 0.5 else 0
        if guess == p['truth']:
            correct += 1

        fpr = false_positives / total_negatives if total_negatives > 0 else 0
        tpr = true_positives / total_positives if total_positives > 0 else 0

        # Every 500 predictions, update the AUC using a trapezoidal approximation.
        if i % 500 == 0:
            if last_fpr is not None:
                trapezoid = (tpr + last_tpr) * (fpr - last_fpr) * 0.5
                auc += trapezoid
            last_fpr = fpr
            last_tpr = tpr

        # In the Lua code, an early break is attempted when recall (tpr) reaches 1.
        if tpr == 1:
            break

    accuracy = correct / len(all_predictions) if all_predictions else 0
    return auc, accuracy

def semi_sorted_mini_batches(dataset, mini_batch_size, trim_to_batch_size):
    """
    Partitions the dataset into mini-batches as follows:
      1. (Optional) Trim the dataset to a multiple of mini_batch_size by shuffling.
      2. Sort the (trimmed) dataset in ascending order by 'n_answers'.
      3. Create mini-batches (each a list of items).
      4. Shuffle the mini-batches.
    
    Args:
      dataset: A list of items (e.g., student records), each with key 'n_answers'.
      mini_batch_size: The desired mini-batch size.
      trim_to_batch_size: If True, trim the dataset so that its length is a multiple
                          of mini_batch_size.
    
    Returns:
      A list of mini-batches (each itself a list).
    """
    if trim_to_batch_size:
        n_temp = len(dataset)
        max_num = n_temp - (n_temp % mini_batch_size)
        # Shuffle indices and select the first max_num items.
        indices = list(range(n_temp))
        random.shuffle(indices)
        trimmed = [dataset[i] for i in indices[:max_num]]
    else:
        trimmed = list(dataset)  # Shallow copy

    # Sort the trimmed dataset by the number of answers.
    trimmed.sort(key=lambda x: x['n_answers'])

    # Partition the dataset into mini-batches.
    mini_batches = [trimmed[i:i + mini_batch_size] for i in range(0, len(trimmed), mini_batch_size)]

    # Shuffle the list of mini-batches.
    indices = list(range(len(mini_batches)))
    random.shuffle(indices)
    shuffled_batches = [mini_batches[i] for i in indices]
    return shuffled_batches
