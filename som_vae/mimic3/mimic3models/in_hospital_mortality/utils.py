from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils
import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}

def load_data_time(reader, discretizer, normalizer, max_seq_len=300, mask_value=0., small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data, time = zip(*[discretizer.transform(X, end=t)[:2] for (X, t) in zip(data, ts)])
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    data = [np.concatenate((item, mask_value + np.zeros((max_seq_len-len(item), item.shape[1])))) 
            if len(item) < max_seq_len else item[:max_seq_len] for item in data]
    time = [np.concatenate((item, mask_value + np.zeros((max_seq_len-len(item))))) 
            if len(item) < max_seq_len else item[:max_seq_len] for item in time]
    whole_data = (np.array(data), np.array(time), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
