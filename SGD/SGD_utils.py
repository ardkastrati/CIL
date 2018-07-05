"""
This script is used to use the SGD model from SGDCollaborativeRegression
"""

import re
import os
import numpy as np
from config import general_params
from config import sgd_params
from SGD.SGDCollaborativeRegressor import SGDCollaborativeRegressor as SGD
MAX_ROW = general_params['n_users']
MAX_COL = general_params['n_movies']
train_data_path = general_params['train_data_path']

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

def extract_training_scores(train_data_path, verbose=False):
    """
    Extract training data from the file.
    :param train_data_path: name of the file to open.
    :param verbose: boolean; if true, it prints information about
    the status of the program.
    :return: X and y. X is a list of tuples (user_id, movie_id); y
    is the list of corresponding scores from X.
    """
    X = []
    y = []
    if verbose: print("Loading data... ")
    data, row_col_search = load_data(datapath=train_data_path)
    if verbose: print("data loaded")
    if verbose: print("Extracting data...")
    # Go through the data and extract the user_id, movie_id and its score
    # NOTE: user_id and movie_id is substracted by 1 to start from 0
    for i in range(len(data)):
        pos_string, score = data[i]
        row_col = row_col_search.search(pos_string)
        d = int(row_col.group(1)) - 1
        n = int(row_col.group(2)) - 1
        X += [(d, n)]
        y += [int(score)]
    if verbose: print("data extracted")
    return X, y

def output_submission(matrix, filename, verbose=False):
    """
    Create output file submission for the Kaggle competition
    :param matrix:
    :param filename: name of the file to submit.
    :param verbose: boolean; if true, it prints information about
    the status of the program.
    """
    # read sample submission
    sample_submission_file_path = general_params['test_data_path']
    if verbose: print("Opening submission file...")
    sample_submission_lines = open(sample_submission_file_path, 'r').readlines()
    if verbose: print("Submission file opened.")
    if verbose: print("Writing the data...")
    # other settings
    row_col_search = re.compile('r([0-9]+)_c([0-9]+)')
    submission_path = os.path.join("submission", filename)
    header = "Id,Prediction\n"

    with open(submission_path, 'w') as w:
        w.write(header)
        # iterate through sample file, fetch the correspnding matrix element
        for line in sample_submission_lines[1:]:
            row_col = row_col_search.search(line)
            r = int(row_col.group(1))
            c = int(row_col.group(2))
            # get prediction
            prediction = str(matrix[r - 1][c - 1])
            # write prediction into file
            new_line = "r{}_c{},{}\n".format(r, c, prediction)
            w.write(new_line)
    if verbose: print("Finished.")

def training_error(X_predict, X, y, verbose=False):
    """
    Calculates RMSE training error
    :param X_predict: numpy matrix of predictions
    :param X: list of (user_id, movie_id) tuples
    :param y: real ratings from the (user_id, movie_id) tuples
    :param verbose: boolean; if true, it prints information about
    the status of the program.
    :return: RMSE between predictions and real ratings
    """
    if verbose: print("Calculating training error...")
    error = 0
    for i in range(len(X)):
        (r, c) = X[i]
        prediction = X_predict[r, c]
        error += (prediction - y[i]) ** 2
    error = np.sqrt(error / (len(X)))
    print(len(X))
    if verbose: print("Training error calculated")
    return error

def sgd():
    """
    Trains the SGD model and creates an output submission.
    """
    X, y = extract_training_scores(train_data_path, verbose=False)
    eta = sgd_params['sgd_eta']
    k = sgd_params['sgd_k']
    reg_factor = sgd_params['sgd_reg']
    num_samples = sgd_params['sgd_n_samples']
    reg = SGD(eta=eta, k=k, reg_factor=reg_factor, num_of_samples=num_samples)
    reg.fit(X, y, verbose=False)
    # print(training_error(reg.X_predict, X, y, verbose=False))
    output_submission(reg.X_predict, 'sgd.csv', verbose=True)

def validate():
    rand_state = np.random.RandomState(0)
    X, y = extract_training_scores(train_data_path, verbose=False)
    rand_state.shuffle(X)
    rand_state.shuffle(y)
    train_pct = general_params['train_pct']
    train_size = int(train_pct * len(y))
    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]
    eta = sgd_params['sgd_eta']
    k = sgd_params['sgd_k']
    reg_factor = sgd_params['sgd_reg']
    num_samples = sgd_params['sgd_n_samples']
    reg = SGD(eta=eta, k=k, reg_factor=reg_factor, num_of_samples=num_samples)
    reg.fit(X_train, y_train, verbose=False)
    predictions = reg.predict(X_val)
    error = rmse(predictions, y_val, verbose=False)
    print(error)

def rmse(predictions, y, verbose=False):
    """
    Calculate RMSE.
    :param predictions: List of predictions
    :param y: Real ratings
    :param verbose:
    :return: RMSE
    """
    if verbose: print("Calculating training error...")
    error = 0
    for i in range(len(y)):
        # (r, c) = X[i]
        prediction = predictions[i]
        error += (prediction - y[i]) ** 2
    error = np.sqrt(error / (len(y)))
    print(len(predictions))
    if verbose: print("Training error calculated")
    return error