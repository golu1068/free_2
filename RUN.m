%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%% AUTHOR     : BERGHOUT TAREK
% %%%%% UNIVERSITY : BATNA 2 university Algeria 
% %%%%% EMAIL      : berghouttarek@gmail
% %%%%% UPDATED    : 14.01.2020 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TD_OSELM :(temporal difference online sequential extreme learning machine)
%%
clear all
clc 
%% Initialize
% generate from a fixed destrebution
rand('state',3);
randn('state',0);

%% load data
load('FD001');  % load dataset (dataset is already prepared)
mini_batch=205; % user desired size of mini-batch
%% divide data 
% the training set is devided into mini-batches according to user desired size of mini-batch
% concerning the test set is already devided according to the number of engine
[xtr,ytr]=devide_blocks(xtr_temp,ytr_temp,mini_batch);
clear mini_batch ytr_temp xtr_temp

%% Training Options {Hyperparameters}

Options.activF='sig';    % Activation function
Options.Neurons=100;     % Number of neurons
Options.lambdaMin=0.95;  % Minimalvalue of forgetting factor
Options.mu=0.98;         % Sensitivity factor controls the speed of convergence of the forgetting factor 
Options.gamma=0.01;      % discounting fctor
Options.C=2;             % regularization parameter (to reduce structural risk)

%% Training and evaluation process
[net]=TD_OSELM(xtr,ytr,xts,yts,Options)
clear Options xtr ytr xts yts

%% preformances
Training_Time = net.Tr_Time
Testing_Time  =net.Ts_Time
Training_RMSE =net.tr_acc       
Testing_RMSE =net.ts_acc 
SCORE = net.S_value      

