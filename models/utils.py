from keras import optimizers
from evaluation.metrics import f1, sensitivity, specificity

def get_optimizer(lr=0.0005):
    adam = optimizers.Adam(lr=lr)
    return adam

def get_loss(task='rlos'):
    if task in ['mort', 'phen', 'dec']:
        return "binary_crossentropy",
    elif task == 'rlos':
        return 'mean_squared_error',
    else:
        print('Invalid task name')
        return False

def get_metrics_to_eval(task='rlos'):
    if task == 'mort':
        return [f1, sensitivity, specificity, 'accuracy']
    elif task == 'rlos':
        return ['mse']
    elif task in ['phen', 'dec']:
        return [f1,'accuracy']
    else:
        print('Invalid task name')
        return False