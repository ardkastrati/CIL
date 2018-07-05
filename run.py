from config import general_params as params
from Surprise.surprise import svd
from Surprise.surprise import svdpp
from Surprise.surprise import nmf
from SGD.SGD_utils import sgd
from BPMRMF.IOHelper import bpmrmf
import time

def main():
    start_time = time.time()
    print("Start time: {}".format(start_time))
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
        svdpp()
    else:
        raise Exception('Please choose one of the following models in the config.py file: '
                        'bpmrmf, sgd, nmf, svd')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()
