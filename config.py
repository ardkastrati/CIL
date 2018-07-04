# configuration used by the training and evaluation scripts
general_params = {}
bpmrmf_params = {}
nmf_params = {}
svd_params = {}
sgd_params = {}
"""
Models:
bpmrmf: Bayesian Probabilistic Matrix Factorization with Mixture Rank Matrix Factorization
sgd: SGD Collaborative Filtering
nmf: Non-Negative Matrix Factorization
svd: Singular Value Decomposition
"""
# general parameters
general_params['model'] = "sgd" # bpmrmf, bpmf, sgd, nmf, svd
general_params['train_data_path'] = "data/data_train.csv"
general_params['test_data_path']= "data/sampleSubmission.csv"
general_params['surprise_train_path'] = "data/data_train_surprise.csv"
general_params['output_file'] = "Mixture_Rank_Bayesian_PMF.csv"
general_params['n_users'] = 10000
general_params['n_movies'] = 1000

# bpmrmf and bpmf common parameters
bpmrmf_params['eval_iters'] = 2
bpmrmf_params['train_pct'] = 1.0
bpmrmf_params['beta'] = 2.0
bpmrmf_params['beta0_user'] = 2.0
bpmrmf_params['beta0_item'] = 2.0
bpmrmf_params['nu0_user'] = None
bpmrmf_params['nu0_item'] = None
bpmrmf_params['mu0_user'] = 0
bpmrmf_params['mu0_item'] = 0
bpmrmf_params['converge'] = 1e-5
bpmrmf_params['max_rating'] = 5.
bpmrmf_params['min_rating'] = 1.
bpmrmf_params['n_features'] = [8,9,10,50] #ranks used in bpmrmf
bpmrmf_params['bpmrmf_alpha'] = 1.

# sgd parameters
sgd_params['sgd_eta'] = 0.001
sgd_params['sgd_k'] = 10
sgd_params['sgd_reg'] = 1e-06
sgd_params['std_n_samples'] = 10000

# nmf parameters
nmf_params['verbose'] = True
nmf_params['biased'] = True
nmf_params['n_epochs'] = 30
nmf_params['n_factors'] = 1
nmf_params['reg_pu'] = 0.3
nmf_params['reg_qi'] = 0.3
nmf_params['reg_bu'] = 0.02
nmf_params['reg_bi'] = 0.02
nmf_params['lr_bu'] = 0.005
nmf_params['lr_bi'] = 0.005
# svd parameters

