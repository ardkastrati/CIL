from config import general_params as params
from config import bpmrmf_params
from Surprise.surprise import svd
from Surprise.surprise import svdpp
from SGD.SGD_utils import validate
from Surprise.surprise import train
from Surprise.surprise import nmf
from SGD.SGD_utils import sgd
from MRMF_BPMF.bpmrmf import BPMRMF
import MRMF_BPMF.IOHelper as IOHelper
from MRMF_BPMF.utils.evaluation import RMSE
import numpy as np
import time

train_data_path = params['train_data_path']
test_data_path = params['test_data_path']
surprise_train_path = params['surprise_train_path']

filename = params['output_file']
D = params['n_users']
N = params['n_movies']
# print("Training data...")
data = IOHelper.numpy_training_data(train_data_path, verbose=True)
# print("Test data...")
test_data = IOHelper.numpy_training_data(test_data_path, verbose=True)

def bpmf():
    rand_state = np.random.RandomState(0)
    rand_state.shuffle(data)
    train_size = int(params['train_pct'] * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]
    #for k in bpmrmf_params['n_features']:
    bpmrmf = BPMRMF(n_user=10000, n_item=1000)
    bpmrmf.fit(train, val, test_data, n_iters=bpmrmf_params['eval_iters'])
        #predictions = bpmrmf.predict(val)
        #rmse = RMSE()

def bpmrmf():
    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    # Model parameters
    n_features = bpmrmf_params['n_features']
    eval_iters = bpmrmf_params['eval_iters']
    train_pct = params['train_pct']
    tau = bpmrmf_params['tau']
    beta = bpmrmf_params['beta']
    beta0_user = bpmrmf_params['beta0_user']
    beta0_item = bpmrmf_params['beta0_item']
    nu0_user = bpmrmf_params['nu0_user']
    nu0_item = bpmrmf_params['nu0_item']
    mu0_user = bpmrmf_params['mu0_user']
    mu0_item = bpmrmf_params['mu0_item']
    max_rating = bpmrmf_params['max_rating']
    min_rating = bpmrmf_params['min_rating']
    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]
    # print("training size: %d" % train.shape[0])
    # print("validation size: %d" % val.shape[0])                                                                                                                                                                         max_rating = None, min_rating = None):
    # print("Rank", n_features)
    bpmrmf = BPMRMF(n_user=n_user, n_item=n_item, n_feature=n_features,
    beta=beta, beta0_user=beta0_user, nu0_user=nu0_user, mu0_user=mu0_user,
    beta0_item=beta0_item, nu0_item=nu0_item, mu0_item=mu0_item, seed=42,
    tau=tau, max_rating = max_rating, min_rating=min_rating)
    bpmrmf.fit(train, val, test_data, n_iters=eval_iters)
    # Create submission file
    IOHelper.numpy_output_submission(bpmrmf.predict(), filename, test_data, verbose=True)

def main():
    start_time = time.time()
    if params['model'] == 'bpmrmf':
        print("Started running BPMRMF. If you want to run other methods please choose another model in the config.py file.")
        bpmrmf()
    elif params['model'] == 'sgd':
        print("Started running SGD. If you want to run other methods please choose another model in the config.py file.")
        sgd()
    elif params['model'] == 'nmf':
        print("Started running NMF. If you want to run other methods please choose another model in the config.py file.")
        nmf()
    elif params['model'] == 'svd':
        print("Started running SVD. If you want to run other methods please choose another model in the config.py file.")
        svd()
    elif params['model'] == 'svdpp':
        print("Started running SVD++. If you want to run other methods please choose another model in the config.py file.")
        train()
    else:
        raise Exception('Please choose one of the following models in the config.py file: '
                        'bpmrmf, sgd, nmf, svd')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()
