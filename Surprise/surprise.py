"""
This file uses the Surprise library to implement SVD and NMF
"""
import re
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise import SVD
import os
from config import svd_params
from config import nmf_params
from config import general_params

def surprise_output_submission(algo, filename, test_path, verbose=False):
    """
    Create the output submission for the Kaggle competition
    :param algo: Fitted algorithm ready for prediction.
    :param filename: Name of the file for submission.
    :param verbose: Print running status.
    :return: Creates a csv file for the predictions.
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
        # iterate through sample file, fetch the corresponding matrix element
        for line in sample_submission_lines[1:]:
            row_col = row_col_search.search(line)
            r = int(row_col.group(1))
            c = int(row_col.group(2))
            prediction = algo.predict(uid = str(r-1), iid=str(c-1)).est
            new_line = "r{}_c{},{}\n".format(r, c, prediction)
            w.write(new_line)
    if verbose: print("Finished.")

def get_training_data():
    """
    Get the training data with the format required by the Surprise algorithms
    """
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(general_params['surprise_train_path'], reader=reader)
    return data.build_full_trainset()

# Fit Non-negative matrix factorization model and create output submission
def nmf():
    train_data = get_training_data()
    algo = NMF(verbose=nmf_params['verbose'],
               biased=nmf_params['biased'],
               n_epochs=nmf_params['n_epochs'],
               n_factors=nmf_params['n_factors'],
               reg_pu=nmf_params['reg_pu'],
               reg_qi=nmf_params['reg_qi'],
               reg_bu=nmf_params['reg_bu'],
               reg_bi=nmf_params['reg_bi'],
               lr_bu=nmf_params['lr_bu'],
               lr_bi=nmf_params['lr_bi'])
    algo.fit(train_data)
    surprise_output_submission(algo, 'nmf.csv', general_params['test_data_path'], verbose=True)

# Fit SVD model and create output submission
def svd():
    train_data = get_training_data()
    algo = SVD()
    algo.fit(train_data)
    surprise_output_submission(algo, 'svd.csv', general_params['test_data_path'], verbose=True)
