from pylab import *

def get_costs(filename, thingname):
    costs = []
    with open (filename, 'r') as f:
        for line in f:
            if line.startswith(thingname):
                costs.append(line.split(':')[1].strip())
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

#unaugmented charptb
#validation 
lstm_baseline = get_costs("/data/lisatmp2/maharajt/speech_project/3LSTM_PTB/log.txt", "dev_bpr")
weight_noise = get_costs("/data/lisatmp2/maharajt/speech_project/rnn_type=lstm_weight_noise=0.075.0/log.txt", "dev_bpr")
norm_stab50 = get_costs("/data/lisatmp2/maharajt/speech_project/norm_cost_coeff=50.0/log.txt", "dev_bpr")
#zoneoutcells95shared = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.95_drop_prob_states=0.95_share_mask.0/log.txt", "dev_bpr")
#zoneoutcells5shared = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.5_share_mask.0/log.txt", "dev_bpr")
#elephant95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.95.0/log.txt", "dev_bpr")
#elephant = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.5.0/log.txt", "dev_bpr")
#zoneoutcells95 = get_costs("/data/lisatmp2/maharajt/speech_project/rnn_type=lstm_drop_prob_cells=0.95.0/log.txt", "dev_bpr")
#zoneoutcells5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5.0/log.txt", "dev_bpr")
#zoneoutstates5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.5.0/log.txt", "dev_bpr")
#zoneoutstates95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.95.0/log.txt", "dev_bpr")
zoneoutcells5states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.95_drop_prob_cells=0.5.1/log.txt", "dev_bpr")
#zoneoutcells3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3.1/log.txt", "dev_bpr")
#zoneoutstates99 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.99.1/log.txt", "dev_bpr")
#elephant3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.3.0/log.txt", "dev_bpr")
#zoneoutcells95states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.95_drop_prob_states=0.95.1/log.txt", "dev_bpr")
#zoneoutcells3states5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.5.0/log.txt", "dev_bpr")
#zoneoutcells5states5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.5.3/log.txt", "dev_bpr")
#zoneoutcells5states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.3.0/log.txt", "dev_bpr")
#zoneoutcells95states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.95_drop_prob_states=0.3.0/log.txt", "dev_bpr")
#zoneoutcells3states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.3.0/log.txt", "dev_bpr")
#zoneoutcells5states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", "dev_bpr")
#zoneoutcells3states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.95.1/log.txt", "dev_bpr")
elephant8 = get_costs("/data/lisatmp4/maharajt/wtfcptb/drop_prob_igates=.8_augment=False.1/log.txt", 'dev_bpr')
stoch_depth = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_stoch_depth.2/log.txt", "dev_bpr")

#training - needs to be divided by log2
training_lstm_baseline = get_costs("/data/lisatmp2/maharajt/speech_project/3LSTM_PTB/log.txt", "training_cost_train")
training_weight_noise = get_costs("/data/lisatmp2/maharajt/speech_project/rnn_type=lstm_weight_noise=0.075.0/log.txt", "training_cost_train")
training_norm_stab50 = get_costs("/data/lisatmp2/maharajt/speech_project/norm_cost_coeff=50.0/log.txt", "training_cost_train")
#training_zoneoutcells95shared = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.95_drop_prob_states=0.95_share_mask.0/log.txt", "training_cost_train")
#training_zoneoutcells5shared = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.5_share_mask.0/log.txt", "training_cost_train")
#training_elephant95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.95.0/log.txt", "training_cost_train")
#training_elephant = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.5.0/log.txt", "training_cost_train")
#training_zoneoutcells95 = get_costs("/data/lisatmp2/maharajt/speech_project/rnn_type=lstm_drop_prob_cells=0.95.0/log.txt", "training_cost_train")
#training_zoneoutcells5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5.0/log.txt", "training_cost_train")
#training_zoneoutstates5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.5.0/log.txt", "training_cost_train")
#training_zoneoutstates95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.95.0/log.txt", "training_cost_train")
training_zoneoutcells5states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.95_drop_prob_cells=0.5.1/log.txt", "training_cost_train")
#training_zoneoutcells3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3.1/log.txt", "training_cost_train")
#training_zoneoutstates99 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.99.1/log.txt", "training_cost_train")
#training_elephant3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.3.0/log.txt", "training_cost_train")
#training_zoneoutcells95states95 = get_costs("/data/lisatmp2/maharajt/speech_project/rnn_type=lstm_drop_prob_cells=0.95_drop_prob_states=0.95./log.txt", "training_cost_train")
#training_zoneoutcells3states5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.5.0/log.txt", "training_cost_train")
#training_zoneoutcells5states5 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.5.3/log.txt", "training_cost_train")
#training_zoneoutcells5states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.3.0/log.txt", "training_cost_train")
#training_zoneoutcells95states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.95_drop_prob_states=0.3.0/log.txt", "training_cost_train")
#training_zoneoutcells3states3 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.3.0/log.txt", "training_cost_train")
#training_zoneoutcells5states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", "training_cost_train")
#training_zoneoutcells3states95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_cells=0.3_drop_prob_states=0.95.1/log.txt", "training_cost_train")

#stoch_depth5 = get_costs("/data/lisatmp2/maharajt/speech_project/stoch_depth=0.5.0/log.txt", "dev_bpr")
#stoch_depth95 = get_costs("/data/lisatmp2/maharajt/speech_project/drop_prob_states=0.95.0/log.txt", "dev_bpr")
#stochdepth5/u/maharajt/Downloads/LSTM_PTB_SD_0.5Cell.txt
#zoneoutcells95states5
#states3
#shared3



to_plot=(lstm_baseline,
weight_noise,
norm_stab50,
stoch_depth,
elephant8,
zoneoutcells5states95
)

names=('Unregularized LSTM',
'Weight noise',
'Norm stabilizer',
'Stochastic depth',
'Recurrent dropout',
'Zoneout',
)


#names=(
#'zoneoutcells95shared',
#'zoneoutcells5shared',
#'zoneoutcells95', 
#'zoneoutcells5',
#'zoneoutstates5',
#'zoneoutstates95',
#'zoneoutcells5states95',
#'zoneoutcells3',
#'zoneoutcells3states5',
#'zoneoutcells5states5',
#'zoneoutcells5states3',
#'zoneoutcells95states3',
#'zoneoutcells3states3',
#'zoneoutcells5states95',
#'zoneoutcells3states95'
#)
#to_plot=(
#zoneoutcells95shared,
#zoneoutcells5shared,
#zoneoutcells95, 
#zoneoutcells5,
#zoneoutstates5,
#zoneoutstates95,
#zoneoutcells5states95,
#zoneoutcells3,
#zoneoutcells3states5,
#zoneoutcells5states5,
#zoneoutcells5states3,
#zoneoutcells95states3,
#zoneoutcells3states3,
#zoneoutcells5states95,
#zoneoutcells3states95
#)
#len(to_plot)

for i in range(len(names)):
    plot(to_plot[i], label=names[i], marker=markers[i])
legend()
show()
#3-5
#- norm_stab50
#- hiddens5
#- hiddens95
#5-7
#- zoneout4
#- zoneout3
#- per_unit5
#7-9
#- zoneout 3/4 shared? (does percentage make no difference?)
#- rnn baseline 
#- rnn bestnum
#- rnn perunit?
#9-11
#- gru baseline
#- gru bestnum
#- gru per unit?
#11-1
#- stoch_depth5
#- stoch_depth95
#1-3
#- per_unit?
#- gaussian?
#- passout?

#*seq_mnist cells only
#*batchnorm



#CHARPTB

#LSTM baseline
#--drop_prob_igates=0.5
#--drop_prob_igates=0.3

#--drop_prob_cells=0.3 --share_mask --stoch_depth
#--drop_prob_cells=0.5 --share_mask --stoch_depth
#--drop_prob_cells=0.95 --share_mask --stoch_depth
#--drop_prob_cells=0.5 --drop_prob_states=0.95 --stoch_depth

#--drop_prob_cells=0.5 --drop_prob_states=0.5
#--drop_prob_cells=0.5 --drop_prob_states=0.3
#--drop_prob_cells=0.3 --drop_prob_states=0.95
#--drop_prob_cells=0.3 --drop_prob_states=0.5
#--drop_prob_cells=0.3 --drop_prob_states=0.3
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob=0.3 --share_mask
#--drop_prob=0.5 --share_mask
#--drop_prob=0.95 --share_mask
#--drop_prob_cells=0.3 --drop_prob_states=0.3
#--drop_prob_cells=0.5 --drop_prob_states=0.5
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob_states=0.3
#--drop_prob_states=0.5
#--drop_prob_states=0.95
#--drop_prob_cells=0.3
#--drop_prob_cells=0.5
#--drop_prob_cells=0.95

#WORDPTB
#lstm baseline
#--weight_noise=0.075
#--norm_cost_coeff=50
#--drop_prob=0.5 --share_mask --stoch_depth
#--drop_prob_cells=0.5 --drop_prob_states=0.95
#--drop_prob_igates=0.3
#--drop_prob_igates=0.5

#CHARPTB
#LSTM baseline
#--drop_prob_igates=0.3
#--drop_prob_cells=0.5 --drop_prob_states=0.95



#--drop_prob_igates=0.95
#--drop_prob_cells=0.3 --share_mask --stoch_depth
#--drop_prob_cells=0.5 --share_mask --stoch_depth
#--drop_prob_cells=0.95 --share_mask --stoch_depth
#--drop_prob_cells=0.5 --drop_prob_states=0.95
#--drop_prob_cells=0.5 --drop_prob_states=0.5
#--drop_prob_cells=0.5 --drop_prob_states=0.3
#--drop_prob_cells=0.3 --drop_prob_states=0.95
#--drop_prob_cells=0.3 --drop_prob_states=0.5
#--drop_prob_cells=0.3 --drop_prob_states=0.3
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob_cells=0.95 --drop_prob_states=0.5
#--drop_prob_cells=0.95 --drop_prob_states=0.3
#--drop_prob_cells=0.3 --share_mask
#--drop_prob_cells=0.5 --share_mask
#--drop_prob_cells=0.95 --share_mask
#--drop_prob_cells=0.3 --drop_prob_states=0.3
#--drop_prob_cells=0.5 --drop_prob_states=0.5
#--drop_prob_cells=0.95 --drop_prob_states=0.95
#--drop_prob_states=0.3
#--drop_prob_states=0.5
#--drop_prob_states=0.95
#--drop_prob_cells=0.3
#--drop_prob_cells=0.5
#--drop_prob_cells=0.95

#WORDPTB
#lstm baseline
#--weight_noise=0.075
#--norm_cost_coeff=50
#--drop_prob_cells=0.5 --drop_prob_states=0.95
#--drop_prob_igates=0.3

#--drop_prob_cells=0.5 --stoch_depth
#--drop_prob_igates=0.5


#GPUS

#eos19 gpu0 - normstab
#eos4 gpu0 states5
#leto07 gpu1 - stochdepth
#leto52 gpu2 - cells5states95
#bart2 gpu0 - elephant5 WORD.py
#bart2 gpu1 - elephant3 WORD.py
#eos22 gpu0 - labelled leto12 LSTM baseline WORD.py
#leto18 gpu0 - weightnoise WORD.py


#leto12 gpu0 - LSTMbaseline CharPTB zoneout_char_ptb_DATA.py
#eos6 gpu0 - elephant3
#leto4 gpu0 - cells5states95

## scriptname: zoneout_char_ptb_DATA.py
## 
##


#figure()
#lrs = [10,3,1,.3,.1,.03,.01,.001,.002,.003]

##load_strs = ["/data/lisatmp2/maharajt/speech_project/learning_rate=" + str(lr) + ".0/log.txt" for lr in lrs]

## ADAM
#adam_strs= ['/data/lisatmp2/maharajt/speech_project/learning_rate=10..0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=3..0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=1..0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.3.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.1.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.03.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.01.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.001.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.002.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.003.0/log.txt']
## MOMENTUM
#momentum_strs= ['/data/lisatmp2/maharajt/speech_project/learning_rate=10._algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=3._algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=1._algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.3_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.1_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.03_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.01_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.001_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.002_algorithm=sgd_momentum.0/log.txt',
 #'/data/lisatmp2/maharajt/speech_project/learning_rate=.003_algorithm=sgd_momentum.0/log.txt']
#lrs= ['10','3','1','0.3','0.1','0.03','0.01','0.001','0.002','0.003']



#def myplot(load_strs, thing_to_plot='dev_bpr', label_strs, markers=None):
    #for i in range(len(label_strs)):
        #try:
            #if markers:
                #plot(get_costs(load_str[i], thing_to_plot), label=label_strs[i], marker=markers[i])
            #else:
                #plot(get_costs(load_str[i], thing_to_plot), label=label_strs[i])
        #except:
            #print load_str, "did not plot"
            #pass



#def get_results(filename, thing='dev_bpr'):
    #with open (filename, 'r') as f:
