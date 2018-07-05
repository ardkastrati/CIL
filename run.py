from config import general_params as params
from config import bpmrmf_params
from Surprise.surprise import svd
from Surprise.surprise import svdpp
from SGD.SGD_utils import validate
from Surprise.surprise import train
from Surprise.surprise import nmf
from SGD.SGD_utils import sgd
from MRMF_BPMF.IOHelper import bpmrmf
import time

def main():
    start_time = time.time()
    print("Start time: {}".format(start_time))
    if params['model'] == 'bpmrmf':
        bpmrmf()
    elif params['model'] == 'sgd':
        sgd()
    elif params['model'] == 'nmf':
        nmf()
    elif params['model'] == 'svd':
        svd()
    elif params['model'] == 'svdpp':
        train()
    else:
        raise Exception('Please choose one of the following models: '
                        'bpmrmf, sgd, nmf, svd')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()
