from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense, Masking
from keras.losses import binary_crossentropy

from mimic3models.keras_models.transformer_block import TransformerBlock, GlobalMaxPooling1D
from mimic3models.keras_models.temporal_encoding import AddPositionalEncoding, TemporalEncoding
from keras.layers.wrappers import TimeDistributed
from mimic3models.keras_utils import ExtendMask, GetTimestep, LastTimestep
from keras.layers.merge import Multiply
    

class Network(Model):
    def __init__(self, dim, ihm_pos, partition, depth=5, num_heads=8, max_seq_len=300,
                 use_time=False, dropout=0.3, input_dim=75, mask_value=0., **kwargs):
        
        print("==> not used params in network class:", kwargs.keys())
        
        self.dim = dim
        self.transformer_depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_time = use_time
        self.transformer_dropout = dropout
        self.input_dim = input_dim
        self.mask_value = mask_value
        
        
        # Input layers and masking
        X = Input(shape=(self.max_seq_len, input_dim), name='X')
        mX = Masking(mask_value=self.mask_value)(X)

        # Masks
        ihm_M = Input(shape=(1,), name='ihm_M')
        
        # Time
        T = Input(shape=(self.max_seq_len,), name='times')
        inputs = [X, T, ihm_M]
        
        
        embedding_layer = Dense(self.dim, activation='relu', name='embedding_layer')
        if use_time:
            temporal_layer = TemporalEncoding(hidden_size=self.dim, name='temporal_embedding')
        else:
            position_layer = AddPositionalEncoding(name='positional_embedding')
        
        
        L = embedding_layer(mX)
        if use_time:
            _T = temporal_layer(T)
            L = Add()([L, _T])
        else:
            L = position_layer(L)

        for i in range(self.transformer_depth):
            L = TransformerBlock(
                name='transformer' + str(i), num_heads=self.num_heads,
                residual_dropout=self.transformer_dropout,
                attention_dropout=self.transformer_dropout,
                use_masking=True,
                vanilla_wiring=True)(L)
        
        # Output modules
        outputs = []

        # ihm output

        # NOTE: masking for ihm prediction works this way:
        #   if ihm_M = 1 then we will calculate an error term
        #   if ihm_M = 0, our prediction will be 0 and as the label
        #   will also be 0 then error_term will be 0.
        ihm_y = Lambda(lambda t: K.expand_dims(K.cast(t <= ihm_pos, dtype='float32')))(T)
        ihm_y = Multiply()([L, ihm_y])
        ihm_y = GlobalMaxPooling1D()(ihm_y)
        ihm_y = Dense(1, activation='sigmoid')(ihm_y)
        ihm_y = Multiply(name='ihm')([ihm_y, ihm_M])
        outputs += [ihm_y]
        
        # decomp output
        decomp_y = GlobalMaxPooling1D()(L)
        decomp_y = Dense(1, activation='sigmoid', name='decomp')(decomp_y)
        outputs += [decomp_y]

        # los output
        los_y = GlobalMaxPooling1D()(L)
        los_y = Dense(1, activation='relu', name='los')(los_y)
        outputs += [los_y]

        # pheno output
        pheno_y = GlobalMaxPooling1D()(L)
        pheno_y = Dense(25, activation='sigmoid', name='pheno')(pheno_y)
        outputs += [pheno_y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}.dep{}".format('transformer',
                                           self.dim,
                                           ".h" if self.num_heads else "",
                                           ".d{}".format(self.transformer_dropout) if self.transformer_dropout > 0 else "",
                                           self.transformer_depth)
    