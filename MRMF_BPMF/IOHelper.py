
import re
import os
import numpy as np
import math
MAX_ROW = 10000
MAX_COL = 1000
datapath = 'data/data_train.csv'

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

    X = []
    y = []

    if verbose: print("Loading data... ")
    data, row_col_search = load_data(datapath=train_data_path)
    if verbose: print("Data loaded")
    if verbose: print("Extracting data...")
    for i in range(len(data)):
        pos_string, score = data[i]
        row_col = row_col_search.search(pos_string)
        d = int(row_col.group(1)) - 1
        n = int(row_col.group(2)) - 1
        X += [(d, n)]
        y += [int(score)]
    if verbose: print("Data extracted")
    return X, y

"""
Same as extract_training_scores but returns a numpy array of shape
(users*items, 3). Each element is a array of (user_id, item_id, score)
"""
def numpy_training_data(train_data_path, verbose=False):

    if verbose: print("Loading data... ")
    data, row_col_search = load_data(datapath=train_data_path)
    if verbose: print("Data loaded")
    if verbose: print("Extracting data...")
    train = np.zeros([len(data), 3], np.int64)
    for i in range(len(data)):
        pos_string, score = data[i]
        row_col = row_col_search.search(pos_string)
        d = int(row_col.group(1)) - 1
        n = int(row_col.group(2)) - 1
        train[i] = np.array([d, n, int(score)])

    if verbose: print("Data extracted")
    return train

def output_submission(matrix, filename, verbose=False):
    # read sample submission
    sample_submission_file_path = "data/sampleSubmission.csv"
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
            prediction = str(matrix[r - 1][c - 1])
            new_line = "r{}_c{},{}\n".format(r, c, prediction)
            w.write(new_line)
    if verbose: print("Finished.")

def numpy_output_submission(preds, filename, implicit_data, verbose=False):
    # read sample submission
    """
    sample_submission_file_path = "data/sampleSubmission.csv"
    if verbose: print("Opening submission file...")
    sample_submission_lines = open(sample_submission_file_path, 'r').readlines()
    if verbose: print("Submission file opened.")
    """

    if verbose: print("Writing the data...")
    # other settings
    row_col_search = re.compile('r([0-9]+)_c([0-9]+)')
    submission_path = os.path.join("submission", filename)
    header = "Id,Prediction\n"

    with open(submission_path, 'w') as w:
        w.write(header)

        # iterate through sample file, fetch the corresponding matrix element
        for i in range(len(implicit_data)):
            new_line = "r{}_c{},{}\n".format(implicit_data[i, 0] + 1, implicit_data[i, 1] + 1, preds[i])
            w.write(new_line)
    if verbose: print("Finished.")

# Surprise output submission
def surprise_output_submission(algo, filename, verbose=False):
    # read sample submission
    sample_submission_file_path = "data/sampleSubmission.csv"
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
            prediction = algo.predict(uid = str(r-1), iid=str(c-1)).est
            new_line = "r{}_c{},{}\n".format(r, c, prediction)
            w.write(new_line)
    if verbose: print("Finished.")

def main():

    b = load_data(datapath)
    print("Hello")

if __name__=='__main__':
    main()
