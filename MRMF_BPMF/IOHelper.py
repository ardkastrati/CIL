from config import general_params
import re
import os
import numpy as np

MAX_ROW = general_params['n_users']
MAX_COL = general_params['n_movies']
datapath = general_params['train_data_path']

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
    sample_submission_file_path = "data/sampleSubmission.csv"
    if verbose: print("Opening submission file...")
    sample_submission_lines = open(sample_submission_file_path, 'r').readlines()
    if verbose: print("Submission file opened.")
    """
    # read sample submission
    if verbose: print("Writing the data...")
    # other settings
    row_col_search = re.compile('r([0-9]+)_c([0-9]+)')
    submission_path = os.path.join("submission", filename)
    header = "Id,Prediction\n"

    with open(submission_path, 'w') as w:
        w.write(header)

        # iterate through sample file, fetch the corresponding matrix element
        for i in range(len(test_data)):
            new_line = "r{}_c{},{}\n".format(test_data[i, 0] + 1, test_data[i, 1] + 1, preds[i])
            w.write(new_line)
    if verbose: print("Finished.")

