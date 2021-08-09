import numpy as np
import gc
from data_extractor.utils import get_data_processors
from sklearn.model_selection import KFold
from evaluation.metrics import get_evaluator
from sklearn.metrics import auc
from data_reader.read_data import get_data
from keras import backend as K
from sys import getsizeof

#
import statsmodels.stats.api as sms

def run_training(config, output_dim, activation):
    extract_data, normalize_data = get_data_processors(config.task)

    evaluate = get_evaluator(config.task)
    df_data = extract_data(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    results = {}
    skf = KFold(n_splits=config.k_fold)

    if config.ann:
        from models.ann import build_network as network
    elif config.lir:
        from models.linear_regression import build_network as network
    elif config.lor:
        from models.logistic_regression import build_network as network
    elif not config.ann and not config.lir and not config.lor:
        from models.bilstm import build_network as network


    

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
        print('Running Fold {}...'.format(fold_id+1))
        train_idx = all_idx[train_idx]
        test_idx  = all_idx[test_idx]

        train, test = normalize_data(config, df_data, train_idx, test_idx)

        train_gen, train_steps, test_gen, test_steps, max_time_step_test = get_data(config, train, test)

        if config.task in ['rlos', 'dec']:
            input_size = 12#200
        else:
            input_size = 200#200

        model = network(config, input_size, output_dim=output_dim, activation=activation)

        history = model.fit_generator(train_gen, steps_per_epoch=25,
                            epochs=config.epochs, verbose=1, shuffle=True)

        probs = []
        labels = []
        for batch in range(test_steps):
            x_test, y_test = next(iter(test_gen))
            batch_probs = model.predict(x_test)
            probs.append(batch_probs)
            labels.append(y_test)

        probs = np.vstack(probs)
        labels = np.vstack(labels)

        results[fold_id] = evaluate(probs, labels, max_time_step_test)
        K.clear_session()
    return results

def train_dec(config):
    
    results = run_training(config, output_dim=1, activation='sigmoid')
    mean_fpr = np.linspace(0,1,100)
    mean_tpr = np.mean([results[k]['intrp'] for k in results], axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std([results[k]['intrp'] for k in results], axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([results[k]['auc'] for k in results])
    ppvs = np.mean([results[k]['ppv'] for k in results])
    npvs = np.mean([results[k]['npv'] for k in results])
    aucprs = np.mean([results[k]['aucpr'] for k in results])
    mccs = np.mean([results[k]['mcc'] for k in results])
    specat90 = np.mean([results[k]['specat90'] for k in results])
    ################################################################################
     # Add Confidence Interval for the reported metrics
    l_auc,h_auc = sms.DescrStatsW([results[k]['auc'] for k in results]).tconfint_mean()
    l_spec,h_spec = sms.DescrStatsW([results[k]['specat90'] for k in results]).tconfint_mean()
    l_PPV,h_PPV = sms.DescrStatsW([results[k]['ppv'] for k in results]).tconfint_mean()
    l_NPV,h_NPV = sms.DescrStatsW([results[k]['npv'] for k in results]).tconfint_mean()
    l_aucpr,h_aucpr = sms.DescrStatsW([results[k]['aucpr'] for k in results]).tconfint_mean()
    l_mcc,h_mcc = sms.DescrStatsW([results[k]['mcc'] for k in results]).tconfint_mean()
    ################################################################################


    print("===========================Decompensation=============================")
    print("Mean AUC: {0:0.3f} +- STD: {1:0.3f}".format(mean_auc,std_auc))
    print("PPV: {0:0.3f}".format(np.mean(ppvs)))
    print("NPV: {0:0.3f}".format(np.mean(npvs)))
    print("AUCPR: {0:0.3f}".format(np.mean(aucprs)))
    print("MCC: {0:0.3f}".format(np.mean(mccs)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90)))

    return {'l_auc':l_auc, 'mean_auc': mean_auc,'h_auc':h_auc,
            'std_auc': std_auc,
            'l_PPV':l_PPV, 'ppv': ppvs,'h_PPV':h_PPV,
            'l_NPV':l_NPV, 'npv': npvs,'h_NPV':h_NPV,
            'l_aucpr':l_aucpr,'aucpr': aucprs,'h_aucpr':h_aucpr,
            'l_mcc':l_mcc,'mcc': mccs,'h_mcc':h_mcc,
            'l_spec':l_spec,'spec@90': specat90,'h_spec':h_spec}

def train_mort(config):
    results = run_training(config, output_dim=1, activation='sigmoid')
    mean_fpr = np.linspace(0,1,100)
    mean_tpr = np.mean([results[k]['intrp'] for k in results], axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std([results[k]['intrp'] for k in results], axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([results[k]['auc'] for k in results])
    ppvs = np.mean([results[k]['ppv'] for k in results])
    npvs = np.mean([results[k]['npv'] for k in results])
    aucprs = np.mean([results[k]['aucpr'] for k in results])
    mccs = np.mean([results[k]['mcc'] for k in results])
    specat90 = np.mean([results[k]['specat90'] for k in results])

    ################################################################################
     # Add Confidence Interval for the reported metrics
    l_auc,h_auc = sms.DescrStatsW([results[k]['auc'] for k in results]).tconfint_mean()
    l_spec,h_spec = sms.DescrStatsW([results[k]['specat90'] for k in results]).tconfint_mean()
    l_PPV,h_PPV = sms.DescrStatsW([results[k]['ppv'] for k in results]).tconfint_mean()
    l_NPV,h_NPV = sms.DescrStatsW([results[k]['npv'] for k in results]).tconfint_mean()
    l_aucpr,h_aucpr = sms.DescrStatsW([results[k]['aucpr'] for k in results]).tconfint_mean()
    l_mcc,h_mcc = sms.DescrStatsW([results[k]['mcc'] for k in results]).tconfint_mean()
    ################################################################################

    print("===========================Mortality=============================")
    print("Mean AUC: {0:0.3f} +- STD: {1:0.3f}".format(mean_auc,std_auc))
    print("PPV: {0:0.3f}".format(np.mean(ppvs)))
    print("NPV: {0:0.3f}".format(np.mean(npvs)))
    print("AUCPR: {0:0.3f}".format(np.mean(aucprs)))
    print("MCC: {0:0.3f}".format(np.mean(mccs)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90)))

    return {'l_auc':l_auc, 'mean_auc': mean_auc,'h_auc':h_auc,
            'std_auc': std_auc,
            'l_PPV':l_PPV, 'ppv': ppvs,'h_PPV':h_PPV,
            'l_NPV':l_NPV, 'npv': npvs,'h_NPV':h_NPV,
            'l_aucpr':l_aucpr,'aucpr': aucprs,'h_aucpr':h_aucpr,
            'l_mcc':l_mcc,'mcc': mccs,'h_mcc':h_mcc,
            'l_spec':l_spec,'spec@90': specat90,'h_spec':h_spec}

def train_phen(config):
    results = run_training(config, output_dim=25, activation='sigmoid')
    aucs = [results[k] for k in results]
    l_auc,h_auc = sms.DescrStatsW([results[k] for k in results]).tconfint_mean()
    aucs_mean = np.mean(np.array(aucs),axis=0)
    aucs_std  =  np.std(np.array(aucs),axis=0)
    for i in range(len(config.col_phe)):
        print("{0} : {1:0.3f} +- {2:0.3f}".format(config.col_phe[i],aucs_mean[i],aucs_std[i]))
    
    
    return {'l_auc':l_auc, 'AUROC mean': aucs_mean,'h_auc':h_auc,'AUROC std': aucs_std}

def train_rlos(config):
    results = run_training(config, output_dim=1, activation='relu')

    print(getsizeof(results))

    meanr2s = np.mean([results[k]['r2'] for k in results])
    meanmses = np.mean([results[k]['mse'] for k in results])
    meanmaes = np.mean([results[k]['mae'] for k in results])

    stdr2s = np.std([results[k]['r2'] for k in results])
    stdmses = np.std([results[k]['mse'] for k in results])
    stdmaes = np.std([results[k]['mae'] for k in results])

    ################################################################################
     # Add Confidence Interval for the reported metrics
    l_r2,h_r2 = sms.DescrStatsW([results[k]['r2'] for k in results]).tconfint_mean()
    l_mae,h_mae = sms.DescrStatsW([results[k]['mae'] for k in results]).tconfint_mean()

    print("===========================RLOS=============================")
    print("R2 total: {0:0.3f} +- {1:0.3f} ".format(meanr2s,stdr2s))
    print("MSE total: {0:0.3f} +- {1:0.3f}".format(meanmses,stdmses))
    print("MAE total: {0:0.3f} +- {1:0.3f}".format(meanmaes,stdmaes))

    return {'R2 L': l_r2,
         'R2 mean': meanr2s,
         'R2 H': h_r2,
         'MAE L':l_mae,
         'MAE mean': meanmaes,
         'MAE H':h_mae,
         'R2 std': stdr2s,
         'MSE mean': meanmses,
         'MSE std': stdmses,
         'MAE std': stdmaes}