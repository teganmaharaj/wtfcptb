import numpy as np
from pylab import *

def get_costs(filename, thingname):
    costs = []
    with open (filename, 'r') as f:
        for line in f:
            if line.startswith(thingname):
                costs.append(float(line.split(':')[1].strip())/np.log(2.))
    return costs

markers=(".",
         ",",
         "o",
         "v",
         "^",
         "<",
         ">",
         "8",
         "s",
         "*",
         "h",
         "H",
         "+",
         "x",
         "D",
         "d",
         "_")

##validation 
#zoneoutcells5states95 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_zoneout5c05s_drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", 'dev_nll_cost')
#elephant25 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant_drop_prob_igates=0.75.0/log.txt", 'dev_nll_cost')
#elephant5 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant5_drop_prob_igates=0.5.0/log.txt", 'dev_nll_cost')
#old_elephant8 = get_costs("/data/lisatmp4/maharajt/wtfcptb/drop_prob_igates=.8_augment=False.1/log.txt", 'dev_nll_cost')

##training - needs to be divided by log2
#zoneoutcells5states95_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_zoneout5c05s_drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", 'train_cost_train')
#elephant25_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant_drop_prob_igates=0.75.0/log.txt", 'train_cost_train')
#elephant5_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant5_drop_prob_igates=0.5.0/log.txt", 'train_cost_train')
##/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.5.1/
#old_elephant8_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/drop_prob_igates=.8_augment=False.1/log.txt", 'train_cost_train')

#to_plot=(
#zoneoutcells5states95,
#elephant25,
#old_elephant7,
#old_elephant8,
##elephant5,
#zoneoutcells5states95_tr,
#elephant25_tr,
#old_elephant5_tr
#old_elephant8_tr
##elephant5_tr,

#old_elephant_tr
#)

#names=('Zoneout z_c=0.5 z_h=0.05',
       #'Recurrent dropout d=0.25',
       #'Recurrent dropout d=0.5',
       #'(t) Zoneout z_c=0.5 z_h=0.05',
       #'(t) Recurrent dropout d=0.25',
       #'(t) Recurrent dropout d=0.5'
#)

#gru_train = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exprnn_type=gru_0/log.txt', 'train_cost_train')
#gru_val = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exprnn_type=gru_0/log.txt', 'dev_nll_cost')
#lstm_train = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exp_4/log.txt', 'train_cost_train')
#lstm_val = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exp_4/log.txt', 'dev_nll_cost')
#srnn_train = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exprnn_type=srnn_0/log.txt', 'train_cost_train')
#srnn_val = get_costs('/data/lisatmp4/maharajt/RNN_memorization/wtfcptb/exprnn_type=srnn_0/log.txt', 'dev_nll_cost')
#to_plot=[lstm_train,lstm_val,gru_train,gru_val,srnn_train,srnn_val]
#names = ['LSTM Training','LSTM Validation', 'GRU Training', 'GRU Validation', 'SRNN Training', 'SRNN Validation']

percent20_train = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=20.1/log.txt', 'train_cost_train')
percent20_val = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=20.1/log.txt', 'dev_nll_cost')
percent40_train = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=40.1/log.txt', 'train_cost_train')
percent40_val = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=40.1/log.txt', 'dev_nll_cost')
percent60_train = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=60.1/log.txt', 'train_cost_train')
percent60_val = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=60.1/log.txt', 'dev_nll_cost')
percent80_train = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=80.1/log.txt', 'train_cost_train')
percent80_val = get_costs('/RQusagers/maharajt/wtfcptb/percent_of_data=80.1/log.txt', 'dev_nll_cost')

to_plot=[percent20_train,
         percent20_val,
         percent40_train,
         percent40_val,
         percent60_train,
         percent60_val,
         percent80_train,
         percent80_val]

names = ['20% Train',
         '20% Val',
         '40% Train',
         '40% Val',
         '60% Train',
         '60% Val',
         '80% Train',
         '80% Val',
         ]

for i in range(len(names)):
    plot(to_plot[i], label=names[i], marker=markers[i])
legend()
show()

