from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3benchmark.readers import DecompensationReader
from mimic3benchmark.readers import LengthOfStayReader
from mimic3benchmark.readers import PhenotypingReader
from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer, OneHotEncoder

import os
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Script for creating a normalizer state - a file which stores the '
                                                 'means and standard deviations of columns of the output of a '
                                                 'discretizer, which are later used to standardize the input of '
                                                 'neural models.')
    parser.add_argument('--task', type=str, required=True,
                        choices=['ihm', 'decomp', 'los', 'pheno', 'multi'])
    parser.add_argument('--impute_strategy', type=str, default='previous',
                        choices=['zero', 'next', 'previous', 'normal_value'],
                        help='Strategy for imputing missing values.')
    parser.add_argument('--store_masks', dest='store_masks', action='store_true',
                        help='Store masks that specify observed/imputed values.')
    parser.add_argument('--no-masks', dest='store_masks', action='store_false',
                        help='Do not store that specify specifying observed/imputed values.')
    parser.add_argument('--n_samples', type=int, default=-1, help='How many samples to use to estimates means and '
                        'standard deviations. Set -1 to use all training samples.')
    parser.add_argument('--output_dir', type=str, help='Directory where the output file will be saved.',
                        default='.')
    parser.add_argument('--data', type=str, required=True, help='Path to the task data.')
    parser.set_defaults(store_masks=True)

    args = parser.parse_args()
    print(args)

    # create the reader
    reader = None
    dataset_dir = os.path.join(args.data, 'train')
    if args.task == 'ihm':
        reader = InHospitalMortalityReader(dataset_dir=dataset_dir, 
                                           listfile=os.path.join(args.data, 'train_listfile.csv'),
                                           period_length=48.0)
    if args.task == 'decomp':
        reader = DecompensationReader(dataset_dir=dataset_dir, listfile=os.path.join(args.data, 'train_listfile.csv'))
    if args.task == 'los':
        reader = LengthOfStayReader(dataset_dir=dataset_dir, listfile=os.path.join(args.data, 'train_listfile.csv'))
    if args.task == 'pheno':
        reader = PhenotypingReader(dataset_dir=dataset_dir, listfile=os.path.join(args.data, 'train_listfile.csv'))
    if args.task == 'multi':
        reader = MultitaskReader(dataset_dir=dataset_dir, listfile=os.path.join(args.data, 'train_listfile.csv'))

    # create the discretizer
    discretizer = OneHotEncoder(impute_strategy=args.impute_strategy)
    discretizer_header = reader.read_example(0)['header']
    continuous_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # create the normalizer
    normalizer = Normalizer(fields=continuous_channels)

    # read all examples and store the state of the normalizer
    n_samples = args.n_samples
    if n_samples == -1:
        n_samples = reader.get_number_of_examples()

    for i in range(n_samples):
        if i % 1000 == 0:
            print('Processed {} / {} samples'.format(i, n_samples), end='\r')
        ret = reader.read_example(i)
        data, _, new_header = discretizer.transform(ret['X'], end=ret['t'])
        normalizer._feed_data(data)
        
    print('\n')

    file_name = '{}_onehotenc_n:{}.normalizer'.format(args.task, n_samples)
    file_name = os.path.join(args.output_dir, file_name)
    print('Saving the state in {} ...'.format(file_name))
    normalizer._save_params(file_name)


if __name__ == '__main__':
    main()