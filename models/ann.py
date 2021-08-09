from __future__ import absolute_import
from __future__ import print_function
import keras
from keras.layers import Dense, Masking, Input, Reshape, Embedding
# from keras import regularizers
# import metrics
from models.utils import get_optimizer, get_loss, get_metrics_to_eval

def build_network(config, input_size, output_dim=1, activation='sigmoid'):
    x_cat = None
    if config.cat:
        if config.ohe:
            input1 = Input(shape=(input_size, config.n_cat_class))
            x_cat = input1
        else:
            input1 = Input(shape=(input_size, 7))
            x1 = Embedding(config.n_cat_class, config.embedding_dim)(input1)
            x_cat = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)

    if config.num:
        if x_cat is not None:
            input2 = Input(shape=(input_size, 13))
            inp = keras.layers.Concatenate(axis=-1)([x_cat, input2])
            inp = Reshape((int(x_cat.shape[1])*int(x_cat.shape[2]+input2.shape[2]),))(inp)
        else:
            input1 = Input(shape=(input_size, 13))
            inp = input1
            inp = Reshape((int(input1.shape[1])*int(input1.shape[2]),))(input1)
    else:
        inp = Reshape((int(x_cat.shape[1])*int(x_cat.shape[2]),))(x_cat)

    mask = Masking(mask_value=0., name="maski")(inp)

    if config.ann:
        hidden = keras.layers.Dense(64,activation='relu')(mask)
    elif not config.ann:
        hidden = mask

    if config.task=='rlos':
        out = Dense(output_dim, activation='relu')(hidden)
    else:
        out = Dense(output_dim, activation='sigmoid')(hidden)


    if config.num and config.cat:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)

    model.compile(loss=get_loss(config.task), optimizer=get_optimizer(lr=config.lr) ,metrics=get_metrics_to_eval(config.task))
    return model
