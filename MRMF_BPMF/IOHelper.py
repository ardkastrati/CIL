from config import general_params as params
from config import bpmrmf_params
from MRMF_BPMF.bpmrmf import BPMRMF
import re
import os
import numpy as np

train_data_path = params['train_data_path']
test_data_path = params['test_data_path']
D = params['n_users']
N = params['n_movies']

def get_data():
    train_data = numpy_training_data(train_data_path, verbose=True)
    test_data = numpy_training_data(test_data_path, verbose=True)
    return train_data, test_data

def load_data(datapath):
    """
    Load the data
    :param datapath: data directory
    :return: tuple of (list of items of form r3_c6 (row 3, column 6), and the rating)
             and a regex object to match and search each rating
    """
    # ignore the header
    data = open(datapath, 'r').readlines()[1:]
    data = [data.strip().split(',') for data in data]
    row_col_search = re.compile('r([0-9]+)_c([0-9]+)')
    return data, row_col_search

def numpy_training_data(train_data_path, verbose=False):
    """
    Extract training data from the file.
    :param train_data_path: name of the file to open.
    :param verbose: boolean; if true, it prints information about
    the status of the program.
    :return: a numpy array of shape (users*items, 3).
    Each element is a array of (user_id, item_id, score).
    """
    if verbose: print("Loading data... ")
    data, row_col_search = load_data(datapath=train_data_path)
    if verbose: print("data loaded")
    if verbose: print("Extracting data...")
    train = np.zeros([len(data), 3], np.int64)
    # Go through the data and extract the user_id, movie_id and its score
    # NOTE: user_id and movie_id is substracted by 1 to start from 0
    for i in range(len(data)):
        pos_string, score = data[i]
        row_col = row_col_search.search(pos_string)
        d = int(row_col.group(1)) - 1
        n = int(row_col.group(2)) - 1
        train[i] = np.array([d, n, int(score)])
    if verbose: print("data extracted")
    return train

def numpy_output_submission(preds, filename, test_data, verbose=False):
    """
    Create output file submission for the Kaggle competition
    :param preds: Numpy array of the predictions.
    :param filename: Name of the file to submit.
    :param test_data: Sample submission file.
    :param verbose: boolean; if true, it prints information about
    the status of the program.
    """
    # read sample submission
    if verbose: print("Writing the data...")
    submission_path = os.path.join("submission", filename)
    header = "Id,Prediction\n"

    with open(submission_path, 'w') as w:
        w.write(header)

        # iterate through sample file, fetch the corresponding matrix element
        for i in range(len(test_data)):
            new_line = "r{}_c{},{}\n".format(test_data[i, 0] + 1, test_data[i, 1] + 1, preds[i])
            w.write(new_line)
    if verbose: print("Finished.")

def bpmrmf():
    data, test_data = get_data()
    rand_state = np.random.RandomState(0)
    train_pct = params['train_pct']
    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]

    bpmrmf = BPMRMF(n_user=params['n_users'], n_item=params['n_movies'],
                    n_feature=bpmrmf_params['n_features'], beta=bpmrmf_params['beta'],
                    beta0_user=bpmrmf_params['beta0_user'], nu0_user=bpmrmf_params['nu0_user'],
                    mu0_user=bpmrmf_params['mu0_user'], beta0_item=bpmrmf_params['beta0_item'],
                    nu0_item=bpmrmf_params['nu0_item'], mu0_item=bpmrmf_params['mu0_item'],
                    seed=42, tau=bpmrmf_params['tau'],
                    max_rating = bpmrmf_params['max_rating'],
                    min_rating=bpmrmf_params['min_rating'])
    bpmrmf.fit(train, val, test_data, n_iters=bpmrmf_params['eval_iters'])
    # Create submission file
    numpy_output_submission(bpmrmf.predictions, 'bpmrmf.csv', test_data, verbose=True)
