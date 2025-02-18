import os
import sys
import time
import random
import gc

# Import the translated modules.
# Adjust these import statements based on your project structure.
from data_assist import DataAssistMatrix
from rnn import RNN
from util_exp import semi_sorted_mini_batches, get_total_tests, evaluate

# Global constants
START_EPOCH = 1
OUTPUT_DIR = '../output/trainRNNAssist/'
LEARNING_RATES = [30, 30, 30, 10, 10, 10, 5, 5, 5]
LEARNING_RATE_REPEATS = 4
MIN_LEARNING_RATE = 1

def get_learning_rate(epoch_index):
    """
    Determines the learning rate based on the current epoch.
    In Lua:
        rateIndex = floor((epochIndex - 1) / LEARNING_RATE_REPEATS) + 1
        if rateIndex <= #LEARNING_RATES then rate = LEARNING_RATES[rateIndex] end
    """
    rate = MIN_LEARNING_RATE
    rate_index = ((epoch_index - 1) // LEARNING_RATE_REPEATS) + 1
    if rate_index <= len(LEARNING_RATES):
        rate = LEARNING_RATES[rate_index - 1]
    return rate

def train_mini_batch(rnn, data, mini_batch_size, file_obj, model_id):
    """
    Trains the model in mini-batches.
    Mimics the Lua loop:
      - For each epoch, a learning rate is set.
      - Data is partitioned into blobs (mini-batches) and for each blob the gradients are computed.
      - Every time a cumulative blob size reaches mini_batch_size, an update is performed.
      - Evaluation is run after each epoch.
      - The model is saved after each epoch.
    """
    print('train')
    epoch_index = START_EPOCH
    blob_size = 50

    while True:
        rate = get_learning_rate(epoch_index)
        start_time = time.time()
        
        # Partition the training data into mini-batches using blob_size.
        mini_batches = semi_sorted_mini_batches(data.getTrainData(), blob_size, True)
        total_tests = get_total_tests(mini_batches)
        gc.collect()

        sum_err = 0.0
        num_tests = 0.0
        done = 0
        rnn.zero_grad(350)  # The argument (350) mimics the Lua call.
        mini_err = 0.0
        mini_tests = 0.0

        for i, batch in enumerate(mini_batches, start=1):
            alpha = blob_size / total_tests if total_tests != 0 else 0
            err, tests, max_norm = rnn.calc_grad(batch, rate, alpha)
            sum_err += err
            num_tests += tests
            gc.collect()
            done += blob_size
            mini_err += err
            mini_tests += tests

            if done % mini_batch_size == 0:
                rnn.update(350, rate)
                rnn.zero_grad(350)
                print('trainMini', i / len(mini_batches), 
                      mini_err / mini_tests if mini_tests != 0 else 0, 
                      sum_err / num_tests if num_tests != 0 else 0, rate)
                mini_err = 0.0
                mini_tests = 0.0

        auc, accuracy = evaluate(rnn, data)
        avg_err = sum_err / num_tests if num_tests != 0 else 0
        outline = "{}\t{}\t{}\t{}\t{}\t{}".format(epoch_index, avg_err, auc, accuracy, rate, time.process_time())
        file_obj.write(outline + "\n")
        file_obj.flush()
        print(outline)

        # Save the model.
        model_save_path = os.path.join(OUTPUT_DIR, 'models', f"{model_id}_{epoch_index}")
        rnn.save(model_save_path)
        epoch_index += 1

def run():
    """
    Main training routine.
      - Expects a command-line argument for fileId.
      - Creates a DataAssistMatrix, builds an RNN, and sets training hyperparameters.
      - Creates output directories and a log file.
      - Begins training by calling train_mini_batch().
    """
    if len(sys.argv) < 2:
        raise Exception("Missing fileId argument.")
    file_id = sys.argv[1]

    # Seed the random number generator.
    random.seed(int(time.time()))

    # Initialize data.
    data = DataAssistMatrix()
    gc.collect()

    # Set training parameters.
    n_hidden = 200
    decay_rate = 1
    init_rate = 30
    mini_batch_size = 100
    dropout_pred = True
    max_grad = 5e-5

    print('n_hidden', n_hidden)
    print('init_rate', init_rate)
    print('decay_rate', decay_rate)
    print('mini_batch_size', mini_batch_size)
    print('dropoutPred', dropout_pred)
    print('maxGrad', max_grad)

    print('making rnn...')
    rnn_model = RNN({
        'dropoutPred': dropout_pred,
        'n_hidden': n_hidden,
        'n_questions': data.n_questions,
        'maxGrad': max_grad,
        'maxSteps': 4290,
        'compressedSensing': True,
        'compressedDim': 100,
        'question_mapping': data.question_mapping  # Pass the mapping here Hongmin
    })
    print('rnn made!')

    # Create output directories.
    os.makedirs('../output', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

    file_path = os.path.join(OUTPUT_DIR, file_id + '.txt')
    with open(file_path, "w") as file_obj:
        file_obj.write('n_hidden,{}\n'.format(n_hidden))
        file_obj.write('init_rate,{}\n'.format(init_rate))
        file_obj.write('decay_rate,{}\n'.format(decay_rate))
        file_obj.write('mini_batch_size,{}\n'.format(mini_batch_size))
        file_obj.write('dropoutPred,{}\n'.format(dropout_pred))
        file_obj.write('maxGrad,{}\n'.format(max_grad))
        file_obj.write('-----\n')
        file_obj.write('i\taverageErr\tauc\ttestPred\trate\tclock\n')
        file_obj.flush()

        train_mini_batch(rnn_model, data, mini_batch_size, file_obj, file_id)

if __name__ == "__main__":
    run()
