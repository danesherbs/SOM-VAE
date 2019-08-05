from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import json
import random
import argparse
import os
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer, OneHotEncoder
from mimic3models import metrics
from mimic3models import keras_utils_time as keras_utils
from mimic3models import common_utils

import tensorflow as tf
# tf.disable_v2_behavior()
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)


# Seed value
# Apparently you may use different seed values at each stage
seed_value= args.seed

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = OneHotEncoder(impute_strategy=args.imputation)
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[2].split(',')

cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'pheno_onehotenc_n:29250.normalizer'
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl
args_dict['input_dim'] = len(discretizer_header)

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "",
                                   ".time{}".format(args.use_time),
                                   ".seed{}".format(args.seed))
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)

# Saving config
config = os.path.join(args.output_dir, 'config')
if not os.path.exists(config):
        os.makedirs(config)
with open(os.path.join(config, model.final_name + ".json"), "w") as h:
  json.dump(args.__dict__, h, sort_keys=True)

# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()


# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


if args.mode == 'train':
    
    # Build data generators

    train_data_gen = utils.BatchGenTime(train_reader, discretizer,
                                        normalizer, args.batch_size,
                                        args.max_seq_len, args.mask_value,
                                        args.small_part, target_repl,
                                        shuffle=True, use_time=args.use_time)
    val_data_gen = utils.BatchGenTime(val_reader, discretizer,
                                      normalizer, args.batch_size,
                                      args.max_seq_len, args.mask_value,
                                      args.small_part, target_repl, 
                                      shuffle=False, use_time=args.use_time)
    
    
    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.PhenotypingMetrics(train_data_gen=train_data_gen,
                                                      val_data_gen=val_data_gen,
                                                      batch_size=args.batch_size,
                                                      use_time=args.use_time,
                                                      verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=n_trained_chunks + args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger],
                        verbose=args.verbose)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    test_data_gen = utils.BatchGenTime(test_reader, discretizer,
                                       normalizer, args.batch_size,
                                       args.max_seq_len, args.mask_value,
                                       args.small_part, target_repl,
                                       shuffle=False, use_time=args.use_time, 
                                       return_names=True)
        

    names = []
    ts = []
    labels = []
    predictions = []
    for i in range(test_data_gen.steps):
        print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
        ret = next(test_data_gen)
        if args.use_time:
            [x, t], y = ret["data"]
        else:
            x, y = ret["data"]
        cur_names = ret["names"]
        cur_ts = ret["ts"]
        x = np.array(x)
        if args.use_time:
            pred = model.predict_on_batch([x, t])
        else:
            pred = model.predict_on_batch(x)
        predictions += list(pred)
        labels += list(y)
        names += list(cur_names)
        ts += list(cur_ts)

    metrics.print_metrics_multilabel(labels, predictions)
    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")

# For katib
print("magic_metric=0.42")
