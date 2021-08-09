from __future__ import absolute_import
from __future__ import print_function
import keras
from keras.layers import BatchNormalization, LSTM, Dropout, Dense, TimeDistributed, Masking, Activation, Input, Reshape, Embedding, Bidirectional
from keras import regularizers
# import metrics
from models.utils import get_optimizer, get_loss, get_metrics_to_eval

def build_network(config, input_size, output_dim=1, activation='sigmoid'):
    inp = None
    if config.cat:
        if config.ohe:
            input1 = Input(shape=(input_size, config.n_cat_class))
            x1 = input1
        else:
            input1 = Input(shape=(input_size, 7))
            x1 = Embedding(config.n_cat_class, config.embedding_dim)(input1)
            x1 = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)
        inp = x1

    if config.num:
        if inp is not None:
            input2 = Input(shape=(input_size, 13))
            inp = keras.layers.Concatenate(axis=-1)([x1, input2])
        else:
            input1 = Input(shape=(input_size, 13))
            inp = input1

    mask = Masking(mask_value=0., name="maski")(inp)

    lstm = mask
    for i in range(config.rnn_layers-1):

        lstm = Bidirectional(LSTM(units=config.rnn_units[i],kernel_regularizer=regularizers.l2(0.01),
                                kernel_initializer='glorot_normal', name="lstm_{}".format(i+1),
                                return_sequences=True))(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(config.dropout)(lstm)

    bilstm = Bidirectional(LSTM(units=config.rnn_units[-1],kernel_regularizer=regularizers.l2(0.01),
                            kernel_initializer='glorot_normal', name="lstm_{}".format(config.rnn_layers),
                            return_sequences=False))(lstm)

    batchnorm = BatchNormalization()(bilstm)
    dropout = Dropout(config.dropout)(batchnorm)

    if config.task=='rlos':
        out = Dense(output_dim, activation='relu')(dropout)
    else:
        out = Dense(output_dim, activation=activation)(dropout)



    if config.num and config.cat:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)

    model.compile(loss=get_loss(config.task), optimizer=get_optimizer(lr=config.lr) ,metrics=get_metrics_to_eval(config.task))

    return model