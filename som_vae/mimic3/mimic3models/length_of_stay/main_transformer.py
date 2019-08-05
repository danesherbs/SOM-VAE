from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import argparse
import os
import imp
import re

from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import LengthOfStayReader

from mimic3models.preprocessing import OneHotEncoder, Normalizer, Discretizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.set_defaults(deep_supervision=False)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--n_samples', type=int, default=2391740)
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

# Build readers, discretizers, normalizers
train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                  listfile=os.path.join(args.data, 'train_listfile.csv'))
val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = OneHotEncoder(impute_strategy=args.imputation)
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[2].split(',')
       
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'los_onehotenc_n:{}.normalizer'.format(args.n_samples)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'los'
args_dict['num_classes'] = (1 if args.partition == 'none' else 10)
args_dict['input_dim'] = len(discretizer_header)



# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}.partition={}{}{}".format("" if not args.deep_supervision else ".dsup",
                                                args.batch_size,
                                                ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                args.timestep,
                                                args.partition,
                                                ".time{}".format(args.use_time),
                                                ".seed{}".format(args.seed))
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_function = 'mean_squared_logarithmic_error'
else:
    loss_function = 'sparse_categorical_crossentropy'
# NOTE: categorical_crossentropy needs one-hot vectors
#       that's why we use sparse_categorical_crossentropy
# NOTE: it is ok to use keras.losses even for (B, T, D) shapes

model.compile(optimizer=optimizer_config,
              loss=loss_function)
model.summary()


# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))

# Load data and prepare generators
# Set number of batches in one epoch
train_nbatches = 2000
val_nbatches = 1000
if args.small_part:
    train_nbatches = 20
    val_nbatches = 20

    
if args.mode == 'train':
    train_data_gen = utils.BatchGenTime(reader=train_reader,
                                        discretizer=discretizer,
                                        normalizer=normalizer,
                                        partition=args.partition,
                                        batch_size=args.batch_size,
                                        steps=train_nbatches,
                                        max_seq_len=args.max_seq_len,
                                        mask_value=args.mask_value,
                                        shuffle=True, 
                                        use_time=args.use_time)

    val_data_gen = utils.BatchGenTime(reader=val_reader,
                                      discretizer=discretizer,
                                      normalizer=normalizer,
                                      partition=args.partition,
                                      batch_size=args.batch_size,
                                      steps=val_nbatches,
                                      max_seq_len=args.max_seq_len,
                                      mask_value=args.mask_value,
                                      shuffle=False,
                                      use_time=args.use_time)

    
    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.LengthOfStayMetrics(train_data_gen=train_data_gen,
                                                       val_data_gen=val_data_gen,
                                                       partition=args.partition,
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

    names = []
    ts = []
    labels = []
    predictions = []

    del train_reader
    del val_reader
    test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'test'),
                                     listfile=os.path.join(args.data, 'test_listfile.csv'))
    
    test_data_gen = utils.BatchGenTime(reader=test_reader,
                                      discretizer=discretizer,
                                      normalizer=normalizer,
                                      partition=args.partition,
                                      batch_size=args.batch_size,
                                      steps=None,
                                      max_seq_len=args.max_seq_len,
                                      mask_value=args.mask_value,
                                      shuffle=False,
                                      use_time=args.use_time,
                                      return_names=True)

    for i in range(test_data_gen.steps):
        print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')

        ret = test_data_gen.next(return_y_true=True)
        if args.use_time:
            ([x, t], y_processed, y) = ret["data"]
        else:
            (x, y_processed, y) = ret["data"]
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

    if args.partition == 'log':
        predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
        metrics.print_metrics_log_bins(labels, predictions)
    if args.partition == 'custom':
        predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
        metrics.print_metrics_custom_bins(labels, predictions)
    if args.partition == 'none':
        metrics.print_metrics_regression(labels, predictions)
        predictions = [x[0] for x in predictions]

    path = os.path.join(os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv")
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
