import numpy as np
import time
import math
######################################################################
def devide_blocks(P,T,mini_batch):
     #P: intputs
    # T: targets
    # mini_batch :mini batche size
    
    # number of trainig data
    nTrainingData=np.shape(P)[0]
    # dividion process 
    c=0;Pn=[];Tn=[];
    for n in np.arange(0, nTrainingData,mini_batch):
        if ((n+mini_batch-1) < nTrainingData):
            c=c+1;
            Pn.append(P[n:(n+mini_batch-1), :])
            Tn.append(T[n:(n+mini_batch-1), :])
    
    return Pn,Tn

def scaledata(datain,minval,maxval):
    dataout = datain - np.min(datain[:])
    dataout = (dataout/np.ptp(dataout[:]))*(maxval-minval);
    dataout = dataout + minval;
    
    return dataout

######################################################################
def TD_OSELM(Trinputs,Trtargets,Tsinputs,Tstargets,Options):
    # TD_OSELM : Temporal Differance Online Sequential Extrem Learning Machine
    
    # Inputs
    # Trinputs : training inputs 
    # Trtargets: training targets
    # Tsinputs : testing inputs
    # Tstargets: testing targets
    # Options  : training Options
    # Outputs  
    # net:the trained network 
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ##### AUTHOR     : BERGHOUT TAREK
    # ##### UNIVERSITY : BATNA 2 university Algeria 
    # ##### EMAIL      : berghouttarek@gmail
    # ##### UPDATED    : 14.01.2020 
    
    # Load Options
    activF=Options['activF'][0];         # Activation function
    Neurons=Options['Neurons'][0];    # Network Architecture
    lambdaMin=Options['lambdaMin'][0]   # minimal value of  forgeting factor
    mu=Options['mu'][0]                 # sensitivity factor
    gamma=Options['gamma'][0]           # Discount factor
    C=Options['C'][0]           # regularization parameter
    # store the first value of forgetting factor
    lambdas = np.array([[0]*1]*1)
    lambdas[0]=lambdaMin;
    # Start training 
    start_time_train= time.time()
    # initialize the model 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load initial mini-batches                                           
    P = Trinputs[0]; 
    T = Trtargets[0];
    # generate input weights
    input_weights=np.random.rand(Neurons,np.shape(P)[1]);         
    # calculating the temporal hidden layer 
    # formula (1) 
    tempH=input_weights*np.transpose(P);                 
    # Activation function
    if (activF.lower() == 'radbas'):
        H = radbas(tempH);
    elif (activF.lower() == 'sig'):
        H = SigActFun(tempH);
    # Save the new features representations  
    Hn=[];
    Hn.append(H); 
    # Calculate beta
    # formula (2) 
    B=np.linalg.pinv(np.transpose(H)) * T; 
    # Calculate the covariance matrix
    # formula (8)
    M = np.linalg.pinv((1/C)+H * np.transpose(H));
    # end  of initialization of the model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # start sequential Learning
    # initialize counters
    # formula (5)
    E=[];
    E.append(T-scaledata((np.transpose(H)*B),min(T),max(T))); # initial TD error
    e=[];
    e.append(np.sqrt(np.mean(E[0])));                   # initial RMSE of TD
    c=0;lambdas=[];                                   # counter of accepted mini_batches
    if np.size(Trinputs)>1:
        for t in range(1, np.size(Trinputs)):
            Pnew = Trinputs[t]
            Tnew = Trtargets[t]
            Hnew_temp=input_weights*np.transpose(Pnew)
            if (activF.lower() == 'radbas'):
                Hn.append(radbas(Hnew_temp))
            elif (activF.lower() == 'sig'):
                Hn.append(SigActFun(Hnew_temp))
            E.append(Tnew-scaledata((np.transpose(Hn[t]-gamma*Hn[t-1]) * B),min(Tnew),max(Tnew)))
            e.append(np.sqrt(np.mean((E[t])**2)))
            lambdaa=lambdaMin
            if (np.sqrt(np.mean((E[t])**2))>np.sqrt(np.mean((E[t-1])**2))):
                c = c+1
                lambdaa=lambdaMin+(1-lambdaMin)*math.exp(-mu*np.sqrt(np.mean((E[t])**2)))
                if (lambdaa<=lambdaMin):
                    lambdaa=lambdaMin
                elif lambdaa>=1:
                    lambdaa=1
            
                K =(M * Hn[t]) *((lambdaa+np.eye(np.shape(Pnew)[0])+np.transpose(Hn[t]-gamma*Hn[t-1]) * M * Hn[t])**(-1))
                M = (1/lambdaa)*(M -( K * np.transpose(Hn[t]-gamma*Hn[t-1]) * M))
                B = B + M * Hn[t] * (E[t])
            lambdas.append(lambdaa)
            
    
    end_time_train=time.time()
    Tr_Time=end_time_train-start_time_train
    # End of the sequential phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Training targets
    Htrain=[];Y_hat=[]
    for i in range(np.size(Trinputs)):   
        # calculate the hidden layer
        Htrain=(input_weights*np.transpose(Trinputs[i]));
        if (activF.lower() == 'radbas'):
            Htrain.append(radbas(Htrain))
        elif (activF.lower() == 'sig'):
            Htrain.append(SigActFun(Htrain))
        # Estimate the targets of training
        Y_hat[i]=scaledata(np.transpose(Htrain)* B,min(Trtargets[i]),max(Trtargets[i]));

    # Training RMSE
    aa, bb,cc,error = Score(Y_hat,Trtargets);
    tr_acc=np.mean(error);
    start_testing_time=time.time();
    # Testing targets
    for i in range(np.size(Tsinputs)):
        Htest=(input_weights*Tsinputs[i])
        
        if (activF.lower() == 'radbas'):
            Htest.append(radbas(Htest))
        elif (activF.lower() == 'sig'):
            Htest.append(SigActFun(Htest))
            
        Yts_hat[i]=scaledata(np.transpose(Htest)* B,min(Tstargets[i]),max(Tstargets[i]))
    
    Ts_Time=time.time() - start_testing_time;
    SCORE,S,d,er=Score(Yts_hat,Tstargets)
    ts_acc=np.mean(er)
    
    net = np.array([(0.0, 0.0, 0.0, 0.0, 0.0)],
              dtype=[('Tr_Time', 'f4'), ('Ts_Time', 'f4'), ('S_value', 'f4'), ('tr_acc', 'f4'), ('ts_acc', 'f4')])
    
    net.Tr_Time=Tr_Time;      # training time
    net.Ts_Time=Ts_Time;      # testing time
    # net.Y=Trtargets;          # original output of training set
    # net.Yts=Tstargets;        # original output for testing set
    # net.Y_hat=Y_hat;          # estimated output for training set
    # net.Yts_hat=Yts_hat;      # estimated output for testing set
    net.S_value=SCORE;        # the score value
    # net.S_vector=S;           # the score vector (for each sample)
    # net.d_vecteur=d;          # the differance vector (for each sample)
    # net.R_vector=er;          # the RMSE vector (for each sample)
    net.tr_acc=tr_acc;        # RMSE of training 
    net.ts_acc=ts_acc;        # RMSE of testing 
    # net.Index=Index_Accepted; # indexes of accepted mini batches
    # net.lambdas=lambdas;      # forgetting factors
    # net.R_TrM=e;              % RMSE of TD of each training mini-batch
    
    return net
    