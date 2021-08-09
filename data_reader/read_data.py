import numpy as np
import random
from keras.utils import to_categorical


def get_nlabels(config):
    if config.task == 'phen':
        n_labels = len(config.col_phe)
    elif config.task in ['dec', 'mort', 'rlos']:
        n_labels = 1
    return n_labels

def get_task_specific_features(config, data, n_cat=7):
    if config.num and config.cat:
        return np.array(data)
    elif config.num:
        return np.array(data[:, :, n_cat:])
    elif  config.cat:
        return np.array(data[:, :, :n_cat])
    else:
        raise ValueError

def get_task_specific_labels(config, data):
    if config.task in ['mort', 'phen']:
        return data[:, 0, :].astype(int)
    else:
        return data.astype(int)

def filter_future_data(config, data, label, nrows, n=12, sw=6):
    temp_data, temp_label, temp_rows = [], [], []
    for i in range(len(data)):
        start = 0
        if nrows[i] >= n:
            for j in range(n, nrows[i], sw):
                if nrows[i]-j >=sw:
                    d = np.zeros((n, data.shape[-1]))
                    d[:j-start, :] = data[i, start:j, :]
                    temp_data.append(d)
                    temp_label.append(label[i, j-1, 0])
                    temp_rows.append(min(j, nrows[i]))
                    start += sw

    temp_data = np.array(temp_data)
    temp_label = np.array(temp_label)

    return temp_data, np.expand_dims(temp_label, axis=-1), temp_rows


def get_data_generator(config, data, label, nrows, train=True):
    data_gen = batch_generator(config, data, label, nrows=nrows, batch_size=config.batch_size)
    steps = np.ceil(len(data)/config.batch_size)
    return data_gen, int(steps)

def get_one_hot(data, n_classes):
    one_hot = np.zeros((data.shape[0], data.shape[1], n_classes), dtype=np.int)
    one_hot = (np.eye(n_classes)[data].sum(2) > 0).astype(int)
    # one_hot = to_categorical(data, num_classes=n_classes)
    return one_hot

def get_data(config, train, test):
    nrows_train = train[1]
    nrows_test = test[1]
    n_labels = get_nlabels(config)

    X_train = train[0][:, :, 1:-n_labels]
    X_test = test[0][:, :, 1:-n_labels]

    Y_train = get_task_specific_labels(config, train[0][:, :, -n_labels:])
    Y_test = get_task_specific_labels(config, test[0][:, :, -n_labels:])

    train_gen, train_steps = get_data_generator(config, X_train, Y_train, nrows=nrows_train)
    test_gen, test_steps = get_data_generator(config, X_test, Y_test, nrows=nrows_test)    

    max_time_step = nrows_test
    return  train_gen, train_steps, test_gen, test_steps, max_time_step

def batch_generator(config, data, labels, nrows=None, batch_size=1024, rng=np.random.RandomState(0), shuffle=True,sample=False):
    if config.task in ['rlos','dec']:
        data, labels, nrows = filter_future_data(config, data, labels, nrows)


    while True:
        if shuffle:
            d = list(zip(data, labels, nrows))
            random.shuffle(d)
            data, labels, nrows = zip(*d)
        data = np.stack(data)
        labels = np.stack(labels)
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i+batch_size]
            y_batch = labels[i:i+batch_size]
            if nrows:
                nrows_batch = np.array(nrows)[i:i+batch_size]

            if config.num and config.cat:
                x_cat = x_batch[:, :, :7].astype(int)
                if config.ohe: x_cat = get_one_hot(x_cat, config.n_cat_class)
                x_num = x_batch[:, :, 7:]
                yield [x_cat, x_num], y_batch

            elif config.cat:
                if config.ohe: x_batch = get_one_hot(x_batch.astype(int), config.n_cat_class)
                yield x_batch, y_batch

            elif config.num:
                yield x_batch, y_batch

    if sample:
        ratio_p = len(pos) / len(X_tr)  # 
        ratio_n = len(neg) / len(X_tr)  # ratio imbalanced
        print(ratio_p, ratio_n)
        while True:
            idx_pos = list(range(pos.shape[0]))
            idx_neg = list(range(neg.shape[0]))
            while (len(idx_pos) + len(idx_neg) > (batch_size*0.2)) and (len(idx_neg)>0) and (len(idx_pos)>0):
                idx = rng.choice(idx_pos, int(batch_size*ratio_p))
                pos_selection = pos[idx]
                idx_pos = list(set(idx_pos) - set(idx))

                idx = rng.choice(idx_neg, int(batch_size*ratio_n))
                neg_selection = neg[idx]
                idx_neg = list(set(idx_neg) - set(idx))

                x_batch = np.concatenate([pos_selection,neg_selection])
                y_batch = np.concatenate([[1]*len(pos_selection),[0]*len(neg_selection)])

                index = list(range(len(x_batch)))
                random.shuffle(index)
                x_batch = x_batch[index]
                y_batch = y_batch[index]

                x_nc = x_batch[:,:,7:]
                x_cat = x_batch[:,:,0:7]
                yield [x_nc, x_cat], y_batch


    