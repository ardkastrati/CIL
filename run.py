from config import general_params as params
from Surprise.surprise import svd
from Surprise.surprise import nmf
from SGD.SGD_utils import sgd

import MRMF_BPMF.bpmf as bpmf
import MRMF_BPMF.bpmrmf as bpmrmf

train_data_path = params['train_data_path']
test_data_path = params['test_data_path']
surprise_train_path = params['surprise_train_path']

filename = params['output_file']
D = params['n_users']
N = params['n_movies']
# print("Training data...")
#data = IOHelper.numpy_training_data(train_data_path, verbose=True)
# print("Test data...")
#test_data = IOHelper.numpy_training_data(test_data_path, verbose=True)

def main():
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
