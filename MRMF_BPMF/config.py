# configuration used by the training and evaluation scripts

params = {}
"""
Models:
bpmrmf: Bayesian Probabilistic Matrix Factorization with Mixture Rank Matrix Factorization
bpmf: Bayesian Probabilistic Matrix Factorization
sgd: SGD Collaborative Filtering
nmf: Non-Negative Matrix Factorization
svd: Singular Value Decomposition
"""
# general parameters
params['model'] = "svd" # bpmrmf, bpmf, sgd, nmf, svd
params['train_data_path'] = "data/data_train.csv"
params['test_data_path']= "data/sampleSubmission.csv"
params['output_file'] = "Mixture_Rank_Bayesian_PMF.csv"
params['n_users'] = 10000
params['n_movies'] = 1000

# bpmrmf and bpmf common parameters
params['eval_iters'] = 21
params['train_pct'] = 1.0
params['beta'] = 2.0
params['beta0_user'] = 2.0
params['beta0_item'] = 2.0
params['nu0_user'] = None
params['nu0_item'] = None
params['mu0_user'] = 0
params['mu0_item'] = 0
params['converge'] = 1e-5
params['max_rating'] = 5.
params['min_rating'] = 1.

# bpmrmf parameters
params['n_features'] = [8,9,10,50] #ranks used in bpmrmf
params['bpmrmf_alpha'] = 1.

# bpmf parameters
params['bpmf_n_features'] = 8
params['bpmf_bias'] = False
params['bpmf_implicit'] = False
params['burn_in'] = 10
params['beta0_a'] = 2.0
params['nu0_a'] = None
params['mu0_a'] = 0.
params['beta0_b'] = 2.0
params['nu0_b'] = None
params['mu0_b'] = 0.
params['beta0_implicit'] = 2.0
params['nu0_implicit'] = None
params['mu0_implicit'] = 0.

# sgd parameters
params['sgd_eta'] = 0.001
params['sgd_k'] = 10
params['sgd_reg'] = 1e-06
params['std_n_samples'] = 10000
# nmf parameters
# svd parameters

