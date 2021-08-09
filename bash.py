from config import Config
from engine import main
import itertools

from keras import backend as K

TASK = ['rlos']#'mort', 'phen', 'rlos','dec'
NUM = [True]
CAT = [True]
OHE = [False]
ANN = [False]
LIR = [True]
LOR = [False]


class build_args():
    pass

def run_model(task):
    l = [NUM, CAT, OHE, ANN,LIR,LOR]
    values = list(itertools.product(*l))
    for val in values:
        nu, ca, oh, an, lir, lor = val
        if not nu and not ca:
            continue
        if an and lir and lor:
            continue
        if not ca and oh:
            continue
        
        K.clear_session()
        args = build_args()
        args.task = task
        args.num = nu
        args.cat = ca
        args.ann = an
        args.ohe = oh
        args.lir = lir
        args.lor = lor
        args.mort_window = 24
        config = Config(args)
        print('{}_num_{}_cat_{}_ohe_{}_ann_{}_lir_{}_lor_{} Started'.format(task, str(nu), str(ca),str(oh), str(an), str(lir), str(lor)))
        output = main(config)

        if output:
            print('{}_num_{}_cat_{}_ohe_{}_ann_{}_lir_{}_lor_{} Finished'.format(task, str(nu), str(ca),str(oh), str(an), str(lir), str(lor)))
        else:
            print('Error in output')
            break

if __name__ == "__main__":
    for tsk in TASK:
        run_model(tsk)
