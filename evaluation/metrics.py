import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import keras.backend as K
import tensorflow as tf


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def get_evaluator(task):
    if task == 'dec':
        return evaluate_decompensation
    elif task =='mort':
        return evaluate_mortality
    elif task == 'phen':
        return evaluate_phenotyping
    elif task == 'rlos':
        return evaluate_rlos

def evaluate_decompensation(y_probs, y_true, ts=None):
    # y_test, probs = remove_padded_data(y_true, y_probs, ts)
    y_test = y_true
    probs = y_probs

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    specat90 = 1-fpr[tpr>=0.90][0]
    intrp = interp(np.linspace(0, 1, 100), fpr, tpr)
    intrp[0] = 0.0
    roc_auc = auc(fpr, tpr)

    TN,FP,FN,TP = confusion_matrix(y_test, probs.round()).ravel()
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    average_precision = average_precision_score(y_test, probs)

    mcc = matthews_corrcoef(y_test, probs.round())
# Add Confidence Interval for the reported metrics
    # l_auc,h_auc = sms.DescrStatsW([auc for each fold]).tconfint_mean()
    # l_spec,h_spec = sms.DescrStatsW([specat90 for each fold]).tconfint_mean()
    # l_PPV,h_PPV = sms.DescrStatsW([PPV for each fold]).tconfint_mean()
    # l_NPV,h_NPV = sms.DescrStatsW([NPV for each fold]).tconfint_mean()
    # l_aucpr,h_aucpr = sms.DescrStatsW([average_precision for each fold]).tconfint_mean()
    # l_mcc,h_mcc = sms.DescrStatsW([mcc for each fold]).tconfint_mean()

    return {'specat90': specat90, 'intrp': intrp,
            'fpr': fpr,
            'tpr': tpr, 'auc': roc_auc,
            'ppv': PPV, 'npv': NPV,
            'aucpr': average_precision,
            'mcc': mcc}
            # l_auc,h_auc,
            # l_spec,h_spec,
            # l_PPV,h_PPV,
            # l_NPV,h_NPV,
            # l_aucpr,h_aucpr,
            # l_mcc,h_mcc

def evaluate_mortality(y_probs, y_true, ts=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specat90 = 1-fpr[tpr>=0.90][0]
    intrp = interp(np.linspace(0, 1, 100), fpr, tpr)
    intrp[0] = 0.0
    roc_auc = auc(fpr, tpr)

    TN,FP,FN,TP = confusion_matrix(y_true, y_probs.round()).ravel()
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    average_precision = average_precision_score(y_true,y_probs)
    mcc = matthews_corrcoef(y_true, y_probs.round())


    # # Add Confidence Interval for the reported metrics
    # l_auc,h_auc = sms.DescrStatsW([roc_auc]).tconfint_mean()
    # l_spec,h_spec = sms.DescrStatsW([specat90]).tconfint_mean()
    # l_PPV,h_PPV = sms.DescrStatsW([PPV]).tconfint_mean()
    # l_NPV,h_NPV = sms.DescrStatsW([NPV]).tconfint_mean()
    # l_aucpr,h_aucpr = sms.DescrStatsW([average_precision]).tconfint_mean()
    # l_mcc,h_mcc = sms.DescrStatsW([mcc]).tconfint_mean()

    return {'specat90': specat90, 'intrp': intrp,
            'fpr': fpr,
            'tpr': tpr, 'auc': roc_auc,
            'ppv': PPV, 'npv': NPV,
            'aucpr': average_precision,
            'mcc': mcc}#,
            # 'l_auc':l_auc,
            # 'h_auc':h_auc,
            # 'l_spec':l_spec,
            # 'h_spec':h_spec,
            # 'l_PPV':l_PPV,'h_PPV':h_PPV,
            # 'l_NPV':l_NPV,'h_NPV':h_NPV,
            # 'l_aucpr':l_aucpr,'h_aucpr':h_aucpr,
            # 'l_mcc':l_mcc,'h_mcc':h_mcc}

def evaluate_phenotyping(y_probs, y_true, ts=None):
    n_classes = 25
    fpr, tpr, roc_auc = {}, {}, {}
    # l_auc,h_auc = {},{}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # l_auc[i],h_auc[i] = sms.DescrStatsW([roc_auc[i] for each fold]).tconfint_mean()

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    auc_value = []
    for i in range(n_classes):
        # auc_value.append(l_auc[i])
        auc_value.append(roc_auc[i])
        # auc_value.append(h_auc[i])
    return auc_value

def evaluate_rlos(y_probs, y_true, ts=None):
    true_stay, pred_stay = [], []
    if ts is None:
        for i,(a,b) in enumerate(zip(np.squeeze(y_true), np.squeeze(y_probs))):
            l = np.squeeze(a).argmin()
            true_stay += list(a[:l])
            pred_stay += list(b[:l])
            e =  b[:l] - a[:l]
    else:
        true_stay = y_true
        pred_stay = y_probs
    r2 = r2_score(true_stay, pred_stay)
    mse = mean_squared_error(true_stay, pred_stay)
    mae = mean_absolute_error(true_stay, pred_stay)
    return {'r2': r2, 'mse': mse, 'mae': mae}

def remove_padded_data(true_label, pred_label, ts=None):
    errors, true_stay, pred_stay = [], [], []
    if ts is None:
        for i,(a,b) in enumerate(zip(np.squeeze(true_label), np.squeeze(pred_label))):
            l = np.squeeze(a).argmin()#nrows_ts[i]
            true_stay += list(a[:l])
            pred_stay += list(b[:l])
            e =  b[:l] - a[:l]
            errors += list(e)
    else:
        for i,(a,b) in enumerate(zip(np.squeeze(true_label), np.squeeze(pred_label))):
            l = ts[i]#nrows_ts[i]
            true_stay += list(a[:l])
            pred_stay += list(b[:l])
            e =  b[:l] - a[:l]
            errors += list(e)

    return np.array(true_stay), np.array(pred_stay)
