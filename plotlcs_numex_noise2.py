from pylab import *
import numpy as np
 
def get_costs(filename, thingname):
    costs = []
    try:
        with open (filename, 'r') as f:
            for line in f:
                if line.startswith(thingname+':'):
               	    if "acc" in thingname:
                        costs.append(float(line.split(':')[1].strip())*100.)
                    else:
                        costs.append(float(line.split(':')[1].strip())/np.log(2.))
            return costs
    except:
        print filename

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
         "_",
         )

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

exps_1000=["state_dim=1000.0",
"state_dim=1000_X_noise=0.25.0",
"state_dim=1000_X_noise=0.25_X_noise_type=seq.0",
"state_dim=1000_X_noise=0.5.0",
"state_dim=1000_X_noise=0.5_X_noise_type=seq.0",
"state_dim=1000_X_noise=0.75.0",
"state_dim=1000_X_noise=0.75_X_noise_type=seq.0",
"state_dim=1000_X_noise=1.0",
"state_dim=1000_Y_noise=0.25.0",
"state_dim=1000_Y_noise=0.5.0",
"state_dim=1000_Y_noise=0.75.0",
"state_dim=1000_Y_noise=1.0"]
exps_2000=["state_dim=2000.0",
"state_dim=2000_X_noise=0.25.0",
"state_dim=2000_X_noise=0.25_X_noise_type=seq.0",
"state_dim=2000_X_noise=0.5.0",
"state_dim=2000_X_noise=0.75.0",
"state_dim=2000_X_noise=0.75_X_noise_type=seq.0",
"state_dim=2000_X_noise=1.0",
"state_dim=2000_X_noise=1_X_noise_type=seq.0",
"state_dim=2000_Y_noise=0.25.0",
"state_dim=2000_Y_noise=0.5.0",
"state_dim=2000_Y_noise=0.75.0",
"state_dim=2000_Y_noise=1.0"]
exps_500=["state_dim=500.0",
"state_dim=500_X_noise=0.25.0",
"state_dim=500_X_noise=0.25_X_noise_type=seq.0",
"state_dim=500_X_noise=0.5.0",
"state_dim=500_X_noise=0.75.0",
"state_dim=500_X_noise=0.75_X_noise_type=seq.0",
"state_dim=500_X_noise=1.0",
"state_dim=500_X_noise=1_X_noise_type=seq.0",
"state_dim=500_Y_noise=0.25.0",
"state_dim=500_Y_noise=0.25_Y_noise_type=seq.0",
"state_dim=500_Y_noise=0.5.0",
"state_dim=500_Y_noise=0.75.0",
"state_dim=500_Y_noise=1.0"]


##validation 
#costs500_val = [get_costs(exp+"/log.txt", 'dev_nll_cost') for exp in exps_500] 
#costs1000_val = [get_costs(exp+"/log.txt", 'dev_nll_cost') for exp in exps_1000] 
#costs2000_val = [get_costs(exp+"/log.txt", 'dev_nll_cost') for exp in exps_2000] 

##training 
#costs500_train = [get_costs(exp+"/log.txt", 'train_cost_train') for exp in exps_500]
#costs1000_train = [get_costs(exp+"/log.txt", 'train_cost_train') for exp in exps_1000]
#costs2000_train = [get_costs(exp+"/log.txt", 'train_cost_train') for exp in exps_2000]

##validation 
#acc500_val = [get_costs(exp+"/log.txt", 'dev_acc') for exp in exps_500]
#acc1000_val = [get_costs(exp+"/log.txt", 'dev_acc') for exp in exps_1000]
#acc2000_val = [get_costs(exp+"/log.txt", 'dev_acc') for exp in exps_2000]

##training  
#acc500_train = [get_costs(exp+"/log.txt", 'train_acc') for exp in exps_500]
#acc1000_train = [get_costs(exp+"/log.txt", 'train_acc') for exp in exps_1000]
#acc2000_train = [get_costs(exp+"/log.txt", 'train_acc') for exp in exps_2000]

#trainval2000 = costs2000_train+costs2000_val
#trainval2000_labels = ['train_'+thing for thing in exps_2000] + ['val_'+thing for thing in exps_2000]

#trainval1000 = costs1000_train+costs1000_val
#trainval1000_labels = ['train_'+thing for thing in exps_1000] + ['val_'+thing for thing in exps_1000]

#trainval500 = costs500_train+costs500_val
#trainval500_labels = ['train_'+thing for thing in exps_500] + ['val_'+thing for thing in exps_500]

#trainval2000 = acc2000_train+acc2000_val
#trainval2000_labels = ['train_'+thing for thing in exps_2000] + ['val_'+thing for thing in exps_2000]

#trainval1000 = acc1000_train+acc1000_val
#trainval1000_labels = ['train_'+thing for thing in exps_1000] + ['val_'+thing for thing in exps_1000]

#trainval500 = acc500_train+acc500_val
#trainval500_labels = ['train_'+thing for thing in exps_500] + ['val_'+thing for thing in exps_500]


#fs = ['full' for i in range(len(costs1000_train))]+['none' for i in range(len(costs1000_val))]
#mk = [filled_markers[i] for i in range(len(costs1000_train))]+[filled_markers[i] for i in range(len(costs1000_val))]

experiments = []
with open ('numex_noise_experiments') as f:
    for line in f:
        experiments.append(line.strip()+'/log.txt')

#dirstr = 
numex197_x = [get_costs(thing, 'dev_acc') for thing in experiments[:6]]
numex197_y = [get_costs(thing, 'dev_acc') for thing in experiments[6:12]]
numex3162_x = [get_costs(thing, 'dev_acc') for thing in experiments[12:18]]
numex3162_y = [get_costs(thing, 'dev_acc') for thing in experiments[18:24]]
numex790_x = [get_costs(thing, 'dev_acc') for thing in experiments[24:30]]
numex790_y = [get_costs(thing, 'dev_acc') for thing in experiments[30:]]

numex197_x_train = [get_costs(thing, 'train_acc') for thing in experiments[:6]]
numex197_y_train = [get_costs(thing, 'train_acc') for thing in experiments[6:12]]
numex3162_x_train = [get_costs(thing, 'train_acc') for thing in experiments[12:18]]
numex3162_y_train = [get_costs(thing, 'train_acc') for thing in experiments[18:24]]
numex790_x_train = [get_costs(thing, 'train_acc') for thing in experiments[24:30]]
numex790_y_train = [get_costs(thing, 'train_acc') for thing in experiments[30:]]

to_plot =  [numex197_x_train, numex197_x]
name =  ['Real', 
        'X noise 0.2',
        'X noise 0.2',
        'X noise 0.2',
        'X noise 0.2',
        'X noise 1',
        ] 
names = [n+' Train' for n in name + [n+' Val' for n in name]
fs =    ['full' for i in range(len(name))] + ['none' for i in range(len(name))]
mk =    [filled_markers[i] for i in range(len(name))] + [filled_markers[i] for i in range(len(name))]

for i in range(len(to_plot)):
    plot(to_plot[i], label=names[i], marker=mk[i]), fillstyle=fs[i])

legend()
show()
