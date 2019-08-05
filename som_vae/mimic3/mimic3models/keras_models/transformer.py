from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense, Masking
from keras.losses import binary_crossentropy

from mimic3models.keras_models.transformer_block import TransformerBlock, GlobalMaxPooling1D
from mimic3models.keras_models.temporal_encoding import AddPositionalEncoding, TemporalEncoding
from keras.layers.wrappers import TimeDistributed
    

class Network(Model):
    def __init__(self, dim, task, depth=5, num_heads=8, max_seq_len=300, use_time=False,
                 dropout=0.3, input_dim=75, mask_value=0., num_classes=1, **kwargs):
        
        print("==> not used params in network class:", kwargs.keys())
        
        self.dim = dim
        self.transformer_depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.transformer_dropout = dropout
        self.input_dim = input_dim
        self.mask_value = mask_value
        
        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        
        embedding_layer = Dense(self.dim, activation='relu', name='embedding_layer')
        if use_time:
            temporal_layer = TemporalEncoding(hidden_size=self.dim, name='temporal_embedding')
        else:
            position_layer = AddPositionalEncoding(name='positional_embedding')
        
        # Input layers and masking
        X = Input(shape=(self.max_seq_len, input_dim), name='X')
        inputs = [X]
        mX = Masking(mask_value=self.mask_value)(X)
        
        L = embedding_layer(mX)
        if use_time:
            T = Input(shape=(self.max_seq_len,), name='times')
            inputs.append(T)
            T = temporal_layer(T)
            L = Add()([L, T])
        else:
            L = position_layer(L)

        for i in range(self.transformer_depth):
            L = TransformerBlock(
                name='transformer' + str(i), num_heads=self.num_heads,
                residual_dropout=self.transformer_dropout,
                attention_dropout=self.transformer_dropout,
                use_masking=True,
                vanilla_wiring=True)(L)
        
        L = GlobalMaxPooling1D()(L)
        y = Dense(num_classes, activation=final_activation)(L)
        outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}.dep{}".format('transformer',
                                           self.dim,
                                           ".h" if self.num_heads else "",
                                           ".d{}".format(self.transformer_dropout) if self.transformer_dropout > 0 else "",
                                           self.transformer_depth)
    