'''Main file for the model training'''
import numpy as np
import os
import sys
import re
import pandas as pd
import pickle
import json
from collections import defaultdict

# load local libraries
import word2vec

def main(args):
    """ Main function
    """

    path_in = args.path_in
    path_out = args.path_out

    # output file path
    if args.path_out is None:
        # please specify a path out
        raise Exception('Please specify an output path for the processed file.')
    
    if args.task == 'word2vec':
        '''This function trains word2vec on the given corpus'''
        path_out = args.path_out
        w2v(
            path_in, 
            path_out
            )

        return

    else:
        raise Exception('task {} not recognized. Run main.py --help for details.'.format(args.task))

"""

-------------------------------------------------------------
Main functions
-------------------------------------------------------------


"""

def w2v(path_in, path_out):
    sys.stdout.write('Task: train word2vec on corpus.\n\nOutput file path:{0}\n\n'.format(path_out)); sys.stdout.flush()
    
    word2vec_trainer = word2vec.TrainWord2Vec(path=path_in, sep=args.sep)

    # tokenize the text 
    word2vec_trainer.tokenize_text()

    # train the model
    word2vec_trainer.train_word2vec()

    # save the model and the df
    word2vec_trainer.store_w2v(path_out)

    return

def forward_nn(path_in, path_out):
    
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'task',
        help='Task to perform - choice: \'word2vec\''
    )

    parser.add_argument(
        'path_in', default=None,
        help='Input file path (csv file)')

    parser.add_argument(
        '-path_out', default=None,
        help='Output file path. Where to write the file with cleaned data.')    

    parser.add_argument(
        '-seed', default=None, type=int, help='Random seed.')

    parser.add_argument(
        '-sep', default=',', help='Separator for the input file.')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(seed=args.seed)

    main(args)