import argparse
import os
import sys
import json
import tensorflow as tf
from keras import backend as K
from config import Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_AFFINITY"] = "disabled"
import warnings
warnings.filterwarnings('ignore')

def main(config):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.Session(config=tf_config)
    K.set_session(session)

    if config.task == 'dec':
        from train import train_dec as train_network
    elif config.task =='mort':
        from train import train_mort as train_network
    elif config.task == 'phen':
        from train import train_phen as train_network
    elif config.task =='rlos':
        if 'train_rlos' not in sys.modules:
                from train import train_rlos as train_network
    else:
        print('Invalid task name')

    result = train_network(config)
    output_file_name = config.save_dir+'{}_{}_{}_{}_{}_{}_{}_{}.json'.format(config.task, str(config.num),  str(config.cat), str(config.ohe),str(config.ann),str(config.lir),str(config.lor), config.mort_window)
    with open(output_file_name, 'w') as f:
        f.write(str(result))

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='mort', type=str, required=False, dest='task')
    parser.add_argument("--num", default=True, type=str, required=False, dest='num')
    parser.add_argument("--cat", default=True, type=str, required=False, dest='cat')
    parser.add_argument("--ann", default=False, type=str, required=False, dest='ann')
    parser.add_argument("--ohe", default=False, type=str, required=False, dest='ohe')
    parser.add_argument("--mort_window", default=24, type=int, required=False, dest='mort_window')

    args = parser.parse_args()
    config = Config(args)
    main(config)
