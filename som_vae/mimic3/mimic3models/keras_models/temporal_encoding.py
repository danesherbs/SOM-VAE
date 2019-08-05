import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)

'''
def temporal_signal(times, hidden_size: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Adaptation of positional encoding to include temporal information
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(times, 2) * K.expand_dims(K.expand_dims(inv_timescales, 0), 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=2)
    return signal
'''

def temporal_signal(times, hidden_size: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Adaptation of positional encoding to include temporal information
    """

    num_timescales = hidden_size
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(times, 2) * K.expand_dims(K.expand_dims(inv_timescales, 0), 0)
    signal = K.sin(scaled_time)
    return signal


class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)
        
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        return inputs + self.signal
    
class TemporalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, hidden_size: int = 16, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.hidden_size = hidden_size
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_size)

    def build(self, input_shape):
        return super().build(input_shape)
    
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, mask=None, **kwargs):
        return temporal_signal(inputs, self.hidden_size, self.min_timescale, self.max_timescale)