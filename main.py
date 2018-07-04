#from MRMF_BPMF.bpmrmf import BPMRMF
#from MRMF_BPMF.bpmf import BPMF
from SGD.SGDCollaborativeRegressor import SGDCollaborativeRegressor
import MRMF_BPMF.utils.IOHelper
import numpy as np
from sklearn.decomposition import NMF
from Surprise.surprise import svd
from Surprise.surprise import nmf
from Surprise.surprise import svd
from config import params


train_data_path = params['train_data_path']
test_data_path = params['test_data_path']

filename = params['output_file']
D = params['n_users']
N = params['n_movies']
print("Training data...")
data = IOHelper.numpy_training_data(train_data_path, verbose=True)
print("Test data...")
test_data = IOHelper.numpy_training_data(test_data_path, verbose=True)

def cross_validate_BPMF():

    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    n_features = [[8,9,10]]
    burn_in = -1
    eval_iters = 40
    rand_state.shuffle(data)

    for curr_n_feature in n_features:
        print("crossvalidating BPMF model...")
        print("Rank", curr_n_feature)

        skf = KFold(n_splits=100, shuffle=True, random_state=rand_state)
        for train_indices, validation_indices in skf.split(data):
            print("Test......")
            X_train = data[train_indices]
            X_validation = data[validation_indices]

            print("training size: %d" % X_train.shape[0])
            print("validation size: %d" % X_validation.shape[0])

            bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=curr_n_feature,max_rating=5., min_rating=1., seed=42, burn_in=burn_in)
            bpmf.fit(X_train, X_validation, test_data, n_iters=eval_iters)

def bpmrmf():
    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    # Model parameters
    n_features = params['n_features']
    eval_iters = params['eval_iters']
    train_pct = params['train_pct']
    alpha = params['bpmrmf_alpha']
    beta = params['beta']
    beta0_user = params['beta0_user']
    beta0_item = params['beta0_item']
    nu0_user = params['nu0_user']
    nu0_item = params['nu0_item']
    mu0_user = params['mu0_user']
    mu0_item = params['mu0_item']
    converge = params['converge']
    max_rating = params['max_rating']
    min_rating = params['min_rating']
    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]
    # print("training size: %d" % train.shape[0])
    # print("validation size: %d" % val.shape[0])                                                                                                                                                                         max_rating = None, min_rating = None):
    # print("Rank", n_features)
    bpmf = BPMRMF(n_user=n_user, n_item=n_item, n_feature=n_features,
    beta=beta, beta0_user=beta0_user, nu0_user=nu0_user, mu0_user=mu0_user,
    beta0_item=beta0_item, nu0_item=nu0_item, mu0_item=mu0_item,
    converge=converge, seed=42, alpha=alpha, max_rating = max_rating, min_rating=min_rating)
    # bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_features, max_rating=5., min_rating=1., seed=42)
    bpmf.fit(train, val, test_data, n_iters=eval_iters)
    # Create submission file
    IOHelper.numpy_output_submission(bpmf.predictions, filename, test_data, verbose=True)

def bpmf():

    implicit_data = np.append(data, IOHelper.numpy_training_data(test_data_path, verbose=True), axis=0)
    rand_state = np.random.RandomState(42)
    n_user = D
    n_item = N
    n_features = params['bpmf_n_features']
    beta = params['beta']
    beta0_user = params['beta0_user']
    nu0_user = params['nu0_user']
    mu0_user = params['mu0_user']
    beta0_item = params['beta0_item']
    nu0_item = params['nu0_item']
    mu0_item = params['mu0_item']
    beta0_a = params['beta0_a']
    nu0_a = params['nu0_a']
    mu0_a = params['mu0_a']
    beta0_b = params['beta0_b']
    nu0_b = params['nu0_b']
    mu0_b = params['mu0_b']
    beta0_implicit = params['beta0_implicit']
    nu0_implicit = params['nu0_implicit']
    mu0_implicit = params['mu0_implicit']
    converge = params['converge']
    max_rating = params['max_rating']
    min_rating = params['min_rating']
    is_bias = params['bpmf_bias']
    is_implicit = params['bpmf_implicit']
    eval_iters = params['eval_iters']
    train_pct = params['train_pct']
    burn_in = params['burn_in']

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
    IOHelper.numpy_output_submission(bpmf, filename, verbose=True)

# MSE
def training_error(X_predict, X, y, verbose=False):
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
    X, y = IOHelper.extract_training_scores(train_data_path, verbose=True)
    eta = params['sgd_eta']
    k = params['sgd_k']
    reg_factor = params['sgd_reg']
    num_samples = params['std_n_samples']
    reg = SGDCollaborativeRegressor(eta=eta, k=k, reg_factor=reg_factor, num_of_samples=num_samples)
    reg.fit(X, y, verbose=True)
    print(training_error(reg.X_predict, X, y, verbose=True))
    IOHelper.output_submission(reg.X_predict, filename, verbose=True)

def final_train_BPMRMF():
    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    n_features = [8,9,10,50]
    eval_iters = 21
    train_pct = 1
    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]

    print("training size: %d" % train.shape[0])
    print("validation size: %d" % val.shape[0])

    print("Rank", n_features)
    bpmf = BPMRMF(n_user=n_user, n_item=n_item, n_feature=n_features, max_rating=5., min_rating=1., seed=42)
    bpmf.fit(train, val, test_data, n_iters=eval_iters)
    IOHelper.numpy_output_submission(bpmf.predictions, filename, test_data, verbose=True)



def final_train_real_BPMRMF():
    rand_state = np.random.RandomState(0)
    n_user = D
    n_item = N
    n_features = [8,9,10,50]
    eval_iters = 21
    train_pct = 1
    rand_state.shuffle(data)
    train_size = int(train_pct * data.shape[0])
    train = data[:train_size]
    val = data[train_size:]

    print("training size: %d" % train.shape[0])
    print("validation size: %d" % val.shape[0])

    print("Rank", n_features)
    bpmrmf = BPMRMF(n_user=n_user, n_item=n_item, n_feature=n_features, max_rating=5., min_rating=1., seed=42)
    bpmrmf.fit(train, val, n_iters=eval_iters)
    IOHelper.numpy_output_submission(bpmrmf.predict(test_data), filename, test_data, verbose=True)



def main():
    #final_train_BPMRMF()
    svd()
    """
    if params['model'] == 'bpmrmf':
        bpmrmf()
    elif params['model'] == 'bpmf':
        """
    """
        is_bias and is_implicit: bpmf with bias and implicit data
        is_bias: bpmf with bias
        else: normal bpmf
        """
    """
        is_bias = params['bpmf_bias']
        is_implicit = params['bpmf_implicit']
        bpmf(bias=is_bias, implicit=is_implicit)
    elif params['model'] == 'sgd':
        sgd()
    elif params['model'] == 'nmf':
        nmf()
    elif params['model'] == 'svd':
        svd()
    else:
        raise Exception('Please choose one of the following models: '
                        'bpmrmf, bpmf, sgd, nmf, svd')
    """
if __name__=='__main__':
    main()
