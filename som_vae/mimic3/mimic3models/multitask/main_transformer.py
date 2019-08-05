from __future__ import absolute_import
from __future__ import print_function

from mimic3models.multitask import utils
from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer, OneHotEncoder
from mimic3models import metrics
from mimic3models import keras_utils_time as keras_utils
from mimic3models import common_utils
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

import mimic3models.in_hospital_mortality.utils as ihm_utils
import mimic3models.decompensation.utils as decomp_utils
import mimic3models.length_of_stay.utils as los_utils
import mimic3models.phenotyping.utils as pheno_utils

import numpy as np
import random
import argparse
import os
import imp
import re

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--partition', type=str, default='none', help="log, custom, none")
parser.add_argument('--ihm_C', type=float, default=1.0)
parser.add_argument('--los_C', type=float, default=1.0)
parser.add_argument('--pheno_C', type=float, default=1.0)
parser.add_argument('--decomp_C', type=float, default=1.0)
parser.add_argument('--data', type=str, help='Path to the data of multitasking',
                    default=os.path.join(os.path.dirname(__file__), '../../data/multitask/'))
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
    args.save_every = 2 ** 30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = MultitaskReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = MultitaskReader(dataset_dir=os.path.join(args.data, 'train'),
                             listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = OneHotEncoder(impute_strategy=args.imputation)
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[2].split(',')

cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'multi_onehotenc_n:29250.normalizer'
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['ihm_pos'] = 48.0
args_dict['target_repl'] = target_repl
args_dict['input_dim'] = len(discretizer_header)

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}{}{}_partition={}_ihm={}_decomp={}_los={}_pheno={}".format(
    args.batch_size,
    ".L1{}".format(args.l1) if args.l1 > 0 else "",
    ".L2{}".format(args.l2) if args.l2 > 0 else "",
    args.timestep,
    ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "",
    ".time{}".format(args.use_time),
    ".seed{}".format(args.seed),
    args.partition,
    args.ihm_C,
    args.decomp_C,
    args.los_C,
    args.pheno_C)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)

# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# Define loss functions

loss_dict = {}
loss_weights = {}

# ihm
loss_dict['ihm'] = 'binary_crossentropy'
loss_weights['ihm'] = args.ihm_C

# decomp
loss_dict['decomp'] = 'binary_crossentropy'
loss_weights['decomp'] = args.decomp_C

# los
if args.partition == 'none':
    # other options are: 'mean_squared_error', 'mean_absolute_percentage_error'
    loss_dict['los'] = 'mean_squared_logarithmic_error'
else:
    loss_dict['los'] = 'sparse_categorical_crossentropy'
loss_weights['los'] = args.los_C

# pheno
loss_dict['pheno'] = 'binary_crossentropy'
loss_weights['pheno'] = args.pheno_C

model.compile(optimizer=optimizer_config,
              loss=loss_dict,
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
    # Build data generators
    train_data_gen = utils.BatchGenTime(reader=train_reader,
                                        discretizer=discretizer,
                                        normalizer=normalizer,
                                        ihm_pos=args_dict['ihm_pos'],
                                        partition=args.partition,
                                        target_repl=target_repl,
                                        batch_size=args.batch_size,
                                        small_part=args.small_part,
                                        shuffle=True,
                                        max_seq_len=args.max_seq_len,
                                        mask_value=args.mask_value,
                                        use_time=args.use_time)
    val_data_gen = utils.BatchGenTime(reader=val_reader,
                                      discretizer=discretizer,
                                      normalizer=normalizer,
                                      ihm_pos=args_dict['ihm_pos'],
                                      partition=args.partition,
                                      target_repl=target_repl,
                                      batch_size=args.batch_size,
                                      small_part=args.small_part,
                                      shuffle=False,
                                      max_seq_len=args.max_seq_len,
                                      mask_value=args.mask_value,
                                      use_time=args.use_time)

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.MultitaskMetrics(train_data_gen=train_data_gen,
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
    # ensure that the code uses test_reader
    del train_reader
    del val_reader

    test_reader = MultitaskReader(dataset_dir=os.path.join(args.data, 'test'),
                                  listfile=os.path.join(args.data, 'test_listfile.csv'))

    test_data_gen = utils.BatchGenTime(reader=test_reader,
                                       discretizer=discretizer,
                                       normalizer=normalizer,
                                       ihm_pos=args_dict['ihm_pos'],
                                       partition=args.partition,
                                       target_repl=target_repl,
                                       shuffle=False,
                                       batch_size=args.batch_size,
                                       small_part=args.small_part,
                                       use_time=args.use_time,
                                       return_names=True)
    ihm_y_true = []
    decomp_y_true = []
    los_y_true = []
    pheno_y_true = []

    ihm_pred = []
    decomp_pred = []
    los_pred = []
    pheno_pred = []

    ihm_names = []
    decomp_names = []
    los_names = []
    pheno_names = []

    decomp_ts = []
    los_ts = []
    pheno_ts = []

    for i in range(test_data_gen.steps):
        print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r')
        ret = test_data_gen.next()
        (X, y) = ret["data"]
        outputs = model.predict(X, batch_size=args.batch_size)

        names = list(ret["names"])
        names_extended = np.array(names).repeat(X[0].shape[1], axis=-1)

        if args.use_time:
            ihm_M = X[2]
        else:
            ihm_M = X[1]

        assert len(outputs) == 4  # no target replication
        (ihm_p, decomp_p, los_p, pheno_p) = outputs
        (ihm_t, decomp_t, los_t, pheno_t) = y

        # ihm
        for (m, t, p, name) in zip(ihm_M.flatten(), ihm_t.flatten(), ihm_p.flatten(), names):
            if np.equal(m, 1):
                ihm_y_true.append(t)
                ihm_pred.append(p)
                ihm_names.append(name)

        # decomp
        for x in ret['decomp_ts']:
            decomp_ts += x
        for (name, t, p) in zip(names_extended.flatten(), decomp_t.flatten(), decomp_p.flatten()):
            decomp_names.append(name)
            decomp_y_true.append(t)
            decomp_pred.append(p)

        # los
        for x in ret['los_ts']:
            los_ts += x
        for (name, t, p) in zip(names_extended.flatten(), los_t.flatten(), los_p.flatten()):
            los_names.append(name)
            los_y_true.append(t)
            los_pred.append(p)

        # pheno
        pheno_names += list(names)
        pheno_ts += list(ret["pheno_ts"])
        for (t, p) in zip(pheno_t.reshape((-1, 25)), pheno_p.reshape((-1, 25))):
            pheno_y_true.append(t)
            pheno_pred.append(p)
    print('\n')

    # ihm
    if args.ihm_C > 0:
        print("\n ================= 48h mortality ================")
        ihm_pred = np.array(ihm_pred)
        ihm_ret = metrics.print_metrics_binary(ihm_y_true, ihm_pred)

    # decomp
    if args.decomp_C > 0:
        print("\n ================ decompensation ================")
        decomp_pred = np.array(decomp_pred)
        decomp_ret = metrics.print_metrics_binary(decomp_y_true, decomp_pred)

    # los
    if args.los_C > 0:
        print("\n ================ length of stay ================")
        if args.partition == 'log':
            los_pred = [metrics.get_estimate_log(x, 10) for x in los_pred]
            los_ret = metrics.print_metrics_log_bins(los_y_true, los_pred)
        if args.partition == 'custom':
            los_pred = [metrics.get_estimate_custom(x, 10) for x in los_pred]
            los_ret = metrics.print_metrics_custom_bins(los_y_true, los_pred)
        if args.partition == 'none':
            los_ret = metrics.print_metrics_regression(los_y_true, los_pred)

    # pheno
    if args.pheno_C > 0:
        print("\n =================== phenotype ==================")
        pheno_pred = np.array(pheno_pred)
        pheno_ret = metrics.print_metrics_multilabel(pheno_y_true, pheno_pred)

    print("Saving the predictions in test_predictions/task directories ...")

    # ihm
    ihm_path = os.path.join(os.path.join(args.output_dir,
                                         "test_predictions/ihm", os.path.basename(args.load_state)) + ".csv")
    ihm_utils.save_results(ihm_names, ihm_pred, ihm_y_true, ihm_path)

    # decomp
    decomp_path = os.path.join(os.path.join(args.output_dir,
                                            "test_predictions/decomp", os.path.basename(args.load_state)) + ".csv")
    decomp_utils.save_results(decomp_names, decomp_ts, decomp_pred, decomp_y_true, decomp_path)

    # los
    los_path = os.path.join(os.path.join(args.output_dir,
                                         "test_predictions/los", os.path.basename(args.load_state)) + ".csv")
    los_utils.save_results(los_names, los_ts, los_pred, los_y_true, los_path)

    # pheno
    pheno_path = os.path.join(os.path.join(args.output_dir,
                                           "test_predictions/pheno", os.path.basename(args.load_state)) + ".csv")
    pheno_utils.save_results(pheno_names, pheno_ts, pheno_pred, pheno_y_true, pheno_path)

else:
    raise ValueError("Wrong value for args.mode")