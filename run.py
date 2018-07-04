from config import general_params as params
from config import bpmrmf_params
from config import bpmf_params
from Surprise.surprise import svd
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
    train_pct = bpmrmf_params['train_pct']
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

def bpmf():

    implicit_data = np.append(data, IOHelper.numpy_training_data(test_data_path, verbose=True), axis=0)
    rand_state = np.random.RandomState(42)
    n_user = D
    n_item = N
    n_features = bpmf_params['bpmf_n_features']
    beta = bpmf_params['beta']
    beta0_user = bpmf_params['beta0_user']
    nu0_user = bpmf_params['nu0_user']
    mu0_user = bpmf_params['mu0_user']
    beta0_item = bpmf_params['beta0_item']
    nu0_item = bpmf_params['nu0_item']
    mu0_item = bpmf_params['mu0_item']
    beta0_a = bpmf_params['beta0_a']
    nu0_a = bpmf_params['nu0_a']
    mu0_a = bpmf_params['mu0_a']
    beta0_b = bpmf_params['beta0_b']
    nu0_b = bpmf_params['nu0_b']
    mu0_b = bpmf_params['mu0_b']
    beta0_implicit = bpmf_params['beta0_implicit']
    nu0_implicit = bpmf_params['nu0_implicit']
    mu0_implicit = bpmf_params['mu0_implicit']
    converge = bpmf_params['converge']
    max_rating = bpmf_params['max_rating']
    min_rating = bpmf_params['min_rating']
    is_bias = bpmf_params['bpmf_bias']
    is_implicit = bpmf_params['bpmf_implicit']
    eval_iters = bpmf_params['eval_iters']
    train_pct = bpmf_params['train_pct']
    burn_in = bpmf_params['burn_in']

    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]

    # print("training size: %d" % train.shape[0])
    # print("validation size: %d" % val.shape[0])
    # print("Rank", n_features)
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_features,
                beta=beta, beta0_user=beta0_user, nu0_user=nu0_user,
                mu0_user=mu0_user, beta0_item=beta0_item, nu0_item=nu0_item,
                mu0_item=mu0_item, bias=is_bias, implicit = is_implicit,
                beta0_a = beta0_a, nu0_a=nu0_a, mu0_a = mu0_a,
                beta0_b = beta0_b, nu0_b = nu0_b, mu0_b = mu0_b,
                beta0_implicit = beta0_implicit, nu0_implicit = nu0_implicit,
                mu0_implicit = mu0_implicit, converge = converge,
                max_rating = max_rating, min_rating = min_rating,
                burn_in = burn_in)
    """
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_features,
                beta=beta,
                max_rating=5., min_rating=1.,
                seed=42, bias=is_bias, implicit=is_implicit, random_init=True, early_stopping=False, burn_in=0)
    """
    bpmf.fit(train, implicit_data, val, n_iters=eval_iters)
    bpmf.predict()
    IOHelper.numpy_output_submission(bpmf.predictions, filename, test_data, verbose=True)
    # IOHelper.numpy_output_submission(bpmf, filename, verbose=True)

def main():
    bpmrmf()
    #final_train_BPMRMF()
    # sp.svd(surprise_train_path, test_data_path)
    # sp.nmf()
    # sgd()

    """
    if params['model'] == 'bpmrmf':
        bpmrmf()
    elif params['model'] == 'bpmf':
        is_bias = params['bpmf_bias']
        is_implicit = params['bpmf_implicit']
        bpmf(bias=is_bias, implicit=is_implicit)
    elif params['model'] == 'sgd':
        sgd.sgd()
    elif params['model'] == 'nmf':
        sp.nmf()
    elif params['model'] == 'svd':
        sp.svd()
    else:
        raise Exception('Please choose one of the following models: '
                        'bpmrmf, bpmf, sgd, nmf, svd')
    """

if __name__=='__main__':
    main()
