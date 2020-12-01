import scipy.io
import pandas as pd
import numpy as np
import random  
from all_function import devide_blocks, scaledata,TD_OSELM, SigActFun, Score
##########################################################
mat = scipy.io.loadmat('FD001.mat')
#mat = {k:v for k, v in mat.items() if k[0] != '_'}
#data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
#data.to_csv("example.csv")

###########################################
# ##### AUTHOR     : BERGHOUT TAREK
# ##### UNIVERSITY : BATNA 2 university Algeria 
# ##### EMAIL      : berghouttarek@gmail
# ##### UPDATED    : 14.01.2020 
############################################
## TD_OSELM :(temporal difference online sequential extreme learning machine)
 
## Initialize
# generate from a fixed destrebution
#rand('state',3);
#randn('state',0);

## load data
#load('FD001');  # load dataset (dataset is already prepared)
mini_batch=205; # user desired size of mini-batch
## divide data 
# the training set is devided into mini-batches according to user desired size of mini-batch
# concerning the test set is already devided according to the number of engine
xtr,ytr =devide_blocks(mat['xtr_temp'],mat['ytr_temp'],mini_batch)
#[xtr,ytr]=devide_blocks(mat['xtr_temp'],ytr_temp,mini_batch);
print('xtr=', np.shape(xtr))
Options = np.array([('activF', 0, 0.0, 0.0, 0.0, 0)],
              dtype=[('activF', 'U10'), ('Neurons', 'i4'), ('lambdaMin', 'f4'), ('mu', 'f4'), ('gamma', 'f4'), ('C', 'i4')])
## Training Options {Hyperparameters}

Options['activF']='sig';    # Activation function
Options['Neurons']=100;     # Number of neurons
Options['lambdaMin']=0.95;  # Minimalvalue of forgetting factor
Options['mu']=0.98;         # Sensitivity factor controls the speed of convergence of the forgetting factor 
Options['gamma']=0.01;      # discounting fctor
Options['C']=2;             # regularization parameter (to reduce structural risk)


## Training and evaluation process
net=TD_OSELM(xtr,ytr,mat['xts'],mat['yts'],Options)
#[net]=TD_OSELM(xtr,ytr,xts,yts,Options)
print(net)
## preformances
Training_Time = net.Tr_Time
Testing_Time  =net.Ts_Time
Training_RMSE =net.tr_acc       
Testing_RMSE =net.ts_acc 
SCORE = net.S_value      

