from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import argparse
import os
import imp
import re

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import OneHotEncoder, Normalizer, Discretizer
from mimic3models import metrics
from mimic3models import keras_utils_time as keras_utils
from mimic3models import common_utils

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--n_samples', type=int, default=14681)
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
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

onehotencoder = OneHotEncoder(impute_strategy='previous', max_seq_len=args.max_seq_len, mask_value=args.mask_value)

onehotencoder_header = onehotencoder.transform(train_reader.read_example(0)["X"])[2].split(',')
cont_channels = [i for (i, x) in enumerate(onehotencoder_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_onehotenc_n:{}.normalizer'.format(args.n_samples)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = onehotencoder_header
args_dict['task'] = 'ihm'
args_dict['input_dim'] = len(onehotencoder_header)

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

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))

if args.mode == 'train':
    
    # Read data
    train_raw = utils.load_data_time(train_reader, onehotencoder, normalizer, args.max_seq_len, args.mask_value, args.small_part)
    val_raw = utils.load_data_time(val_reader, onehotencoder, normalizer, args.max_seq_len, args.mask_value, args.small_part)


    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[2])  # (B,)
            data[2] = [labels, None]
            data[2][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[2][1] = np.expand_dims(data[2][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=([train_raw[0],train_raw[1]], train_raw[2]),
                                                              val_data=([val_raw[0], val_raw[1]], val_raw[2]),
                                                              target_repl=(args.target_repl_coef > 0),
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
    if args.use_time:
        model.fit(x=[train_raw[0], train_raw[1]],
                  y=train_raw[2],
                  validation_data=([val_raw[0], val_raw[1]], val_raw[2]),
                  epochs=n_trained_chunks + args.epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=args.verbose,
                  batch_size=args.batch_size)
    else:
        model.fit(x=train_raw[0],
                  y=train_raw[2],
                  validation_data=(val_raw[0], val_raw[2]),
                  epochs=n_trained_chunks + args.epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=args.verbose,
                  batch_size=args.batch_size)


if args.mode == 'test':
    # ensure that the code uses test_reader

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data_time(test_reader, onehotencoder, normalizer, args.max_seq_len, args.mask_value, 
                               args.small_part, return_names=True)

    data = ret["data"][0]
    time = ret["data"][1]
    labels = ret["data"][2]
    names = ret["names"]

    if args.use_time:
        predictions = model.predict([data, time], batch_size=args.batch_size, verbose=1)
    else:
        predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)
