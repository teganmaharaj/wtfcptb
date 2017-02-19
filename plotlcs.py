import numpy as np
from 
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

#validation 
zoneoutcells5states95 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_zoneout5c05s_drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", 'dev_nll_cost')
elephant25 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant_drop_prob_igates=0.75.0/log.txt", 'dev_nll_cost')
elephant5 = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant5_drop_prob_igates=0.5.0/log.txt", 'dev_nll_cost')
old_elephant8 = get_costs("/data/lisatmp4/maharajt/wtfcptb/drop_prob_igates=.8_augment=False.1/log.txt", 'dev_nll_cost')

#training - needs to be divided by log2
zoneoutcells5states95_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_zoneout5c05s_drop_prob_cells=0.5_drop_prob_states=0.95.0/log.txt", 'train_cost_train')
elephant25_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant_drop_prob_igates=0.75.0/log.txt", 'train_cost_train')
elephant5_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/experiment_path=REBUTTAL_OLDCODE_elephant5_drop_prob_igates=0.5.0/log.txt", 'train_cost_train')
#/data/lisatmp2/maharajt/speech_project/drop_prob_igates=0.5.1/
old_elephant8_tr = get_costs("/data/lisatmp4/maharajt/wtfcptb/drop_prob_igates=.8_augment=False.1/log.txt", 'train_cost_train')

to_plot=(
zoneoutcells5states95,
elephant25,
old_elephant7,
old_elephant8,
#elephant5,
zoneoutcells5states95_tr,
elephant25_tr,
old_elephant5_tr
old_elephant8_tr
#elephant5_tr,

old_elephant_tr
)

names=('Zoneout z_c=0.5 z_h=0.05',
       'Recurrent dropout d=0.25',
       'Recurrent dropout d=0.5',
       '(t) Zoneout z_c=0.5 z_h=0.05',
       '(t) Recurrent dropout d=0.25',
       '(t) Recurrent dropout d=0.5'
)


for i in range(len(names)):
    plot(to_plot[i], label=names[i], marker=markers[i])


