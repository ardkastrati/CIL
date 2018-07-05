from config import general_params as params
from config import bpmrmf_params
from Surprise.surprise import svd
from SGD.SGD_utils import validate
from Surprise.surprise import train
from Surprise.surprise import nmf
from SGD.SGD_utils import sgd
from MRMF_BPMF.bpmrmf import BPMRMF
import MRMF_BPMF.IOHelper as IOHelper
import numpy as np

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

def bpmrmf():
    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    # Model parameters
    n_features = bpmrmf_params['n_features']
    eval_iters = bpmrmf_params['eval_iters']
    train_pct = params['train_pct']
    alpha = bpmrmf_params['bpmrmf_alpha']
    beta = bpmrmf_params['beta']
    beta0_user = bpmrmf_params['beta0_user']
    beta0_item = bpmrmf_params['beta0_item']
    nu0_user = bpmrmf_params['nu0_user']
    nu0_item = bpmrmf_params['nu0_item']
    mu0_user = bpmrmf_params['mu0_user']
    mu0_item = bpmrmf_params['mu0_item']
    converge = bpmrmf_params['converge']
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
    beta0_item=beta0_item, nu0_item=nu0_item, mu0_item=mu0_item,
    converge=converge, seed=42, alpha=alpha, max_rating = max_rating, min_rating=min_rating)
    bpmrmf.fit(train, val, test_data, n_iters=eval_iters)
    # Create submission file
    IOHelper.numpy_output_submission(bpmrmf.predictions, filename, test_data, verbose=True)

def main():
    validate()
    """
        if params['model'] == 'bpmrmf':
        bpmrmf()
    elif params['model'] == 'sgd':
        sgd()
    elif params['model'] == 'nmf':
        nmf()
    elif params['model'] == 'svd':
        svd()
    else:
        raise Exception('Please choose one of the following models: '
                        'bpmrmf, sgd, nmf, svd')

    :return:
    """

if __name__=='__main__':
    main()
