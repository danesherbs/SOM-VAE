import argparse
import json
import subprocess
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, help='Hidden dimensionality.',
                    required=True)
parser.add_argument('--dropout', type=float, help='Dropout probability.',
                    required=True)
parser.add_argument('--heads', type=int, help='Number of heads.',
                    required=True)
parser.add_argument('--data', type=str, help='Data dir.',
                    default="/mnt/data/mimic3/phenotyping")
parser.add_argument('--output_dir', type=str, help='Output dir.',
                    default="/mnt/results/mimic3/phenotyping")
args = parser.parse_args()

# print(json.dumps(args.__dict__, sort_keys=True))

depth = 256 // args.dim

COMMON_CMDS = "--network mimic3models/keras_models/transformer.py " \
              "--dim {dim} " \
              "--depth {depth} " \
              "--dropout {dropout} " \
              "--mode train " \
              "--batch_size 8 " \
              "--num_heads {heads} " \
              "--output_dir {output_dir} " \
              "--verbose 1 " \
              "--epochs 50 " \
              "--max_seq_len 2000 " \
              "--mask_value 0. " \
              "--seed 0 " \
              "--data {data} "
COMMON_CMDS = COMMON_CMDS.format(
    **{"dim": args.dim,
       "depth": depth,
       "dropout": args.dropout,
       "heads": args.heads,
       "output_dir": args.output_dir,
       "data": args.data})

DISCRETISED_CMD = "python -um mimic3models.phenotyping.main --timestep 1.0 "
DISCRETISED_CMD += COMMON_CMDS
DISCRETISED_CMD = DISCRETISED_CMD

TIME_FALSE_CMD = "python -um mimic3models.phenotyping.main_transformer "
TIME_FALSE_CMD += COMMON_CMDS

TIME_TRUE_CMD = TIME_FALSE_CMD
TIME_TRUE_CMD += "--use_time True"

# STARS = "*" * 20
#
# pprint(STARS + " Discretised command " + STARS)
# print(DISCRETISED_CMD)

p = subprocess.Popen(
    [x for x in DISCRETISED_CMD.split(" ") if x != " "],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()

# pprint(STARS + " Time false command " + STARS)
# print(TIME_FALSE_CMD)

p = subprocess.Popen([x for x in TIME_FALSE_CMD.split(" ") if x != " "],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()

# pprint(STARS + " Time true command " + STARS)
# print(TIME_TRUE_CMD)

p = subprocess.Popen([x for x in TIME_TRUE_CMD.split(" ") if x != " "],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
