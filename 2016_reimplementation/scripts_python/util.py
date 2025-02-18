import torch
import math
import random
import copy
import csv
import gc

def self_dot(m):
    """
    Returns the dot product of m with itself.
    (Assumes m is a 1D tensor; for matrices use torch.sum(m * m) for the Frobenius norm squared.)
    """
    return torch.dot(m, m)

def table_to_vec(grad_list):
    """
    Concatenates a list of tensors (gradients) along dimension 0.
    """
    return torch.cat(grad_list, dim=0)

def save_matrix_as_csv(path, matrix):
    """
    Saves a 2D tensor as a CSV file.
    """
    # Convert tensor to a list of lists; each element is converted to a Python number.
    mat_list = [[elem.item() for elem in row] for row in matrix]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mat_list)

def split(s, sep=":"):
    """
    Splits the string s by the separator sep.
    """
    return s.split(sep)

def save_matrix_as_gephi(path, matrix):
    """
    Saves a 2D tensor to a file in a format that Gephi can import.
    The first line is a header with column indices, and each subsequent
    line starts with the row index followed by the row values.
    """
    with open(path, "w", newline="") as f:
        # Write header: ;1;2;3;...
        header = ''.join(';' + str(c + 1) for c in range(matrix.size(1)))
        f.write(header + '\n')
        # Write each row (using 1-indexing for rows to mimic Lua)
        for r in range(matrix.size(0)):
            line = str(r + 1)
            for c in range(matrix.size(1)):
                line += ';' + str(matrix[r, c].item())
            f.write(line + '\n')

def copy_without_row(m, index):
    """
    Returns a copy of 2D tensor m without the row at the given index.
    Here, index is assumed to be 0-based.
    """
    n_rows = m.size(0)
    if index == 0:
        return m[1:].clone()
    elif index == n_rows - 1:
        return m[:-1].clone()
    else:
        return torch.cat([m[:index], m[index + 1:]], dim=0)

def get_keyset(d):
    """
    Given a dictionary d, returns a list of its keys.
    """
    return list(d.keys())

def mask_rows(m, y):
    """
    Returns the rows of tensor m for which the corresponding entry in y is nonzero.
    Also prints each value in y.
    """
    for i in range(y.size(0)):
        print(y[i].item())
    # Boolean indexing: select rows where y != 0.
    return m[y != 0]

def shuffle_list(a):
    """
    Returns a shuffled copy of the list a.
    """
    a_copy = a[:]  # shallow copy
    random.shuffle(a_copy)
    return a_copy

def get_param_lin_scale(min_val, max_val):
    """
    Returns a random value linearly scaled between min_val and max_val.
    """
    return random.uniform(min_val, max_val)

def get_param_log_scale(min_val, max_val):
    """
    Returns a random value scaled logarithmically between min_val and max_val.
    """
    x = random.random()
    return min_val * math.exp(math.log(max_val / min_val) * x)

def clone_once(net):
    """
    Clones a PyTorch model by deep copying its structure,
    then sets its parameters to share the same underlying data as the original.
    This mimics the behavior of cloning in the Lua/Torch code.
    """
    # Deep copy the network structure
    clone = copy.deepcopy(net)
    # Share parameters: make sure the clone's parameters point to the same data.
    for orig_param, clone_param in zip(net.parameters(), clone.parameters()):
        clone_param.data = orig_param.data
    # Share any registered buffers (e.g., running means in BatchNorm)
    for orig_buf, clone_buf in zip(net.buffers(), clone.buffers()):
        clone_buf.data = orig_buf.data
    return clone

def clone_many_times(net, T):
    """
    Creates T clones of the network `net`, each sharing parameters with the original.
    """
    clones = []
    # Pre-serialize the network structure if desired (omitted here for simplicity).
    for t in range(T):
        clone = clone_once(net)
        clones.append(clone)
        print('clone', t + 1)
    gc.collect()
    return clones

def run():
    """
    A simple run function that creates a random 5x5 matrix,
    a mask vector y with two nonzero entries, and prints the masked rows.
    """
    a = torch.randn(5, 5)
    y = torch.zeros(5)
    # In Lua, y[2] and y[3] are set to 1 (Lua is 1-indexed).
    # In Python (0-indexed), we set y[1] and y[2] to 1.
    y[1] = 1
    y[2] = 1
    m = mask_rows(a, y)
    print("Matrix a:")
    print(a)
    print("Masked matrix m:")
    print(m)

if __name__ == "__main__":
    run()
