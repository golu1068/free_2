function [net]=TD_OSELM(Trinputs,Trtargets,Tsinputs,Tstargets,Options)
% TD_OSELM : Temporal Differance Online Sequential Extrem Learning Machine

% Inputs
% Trinputs : training inputs 
% Trtargets: training targets
% Tsinputs : testing inputs
% Tstargets: testing targets
% Options  : training Options
% Outputs  
% net:the trained network 

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% %%%%% AUTHOR     : BERGHOUT TAREK
% %%%%% UNIVERSITY : BATNA 2 university Algeria 
% %%%%% EMAIL      : berghouttarek@gmail
% %%%%% UPDATED    : 14.01.2020 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
%
%
%
%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Load Options
activF=Options.activF;         % Activation function
Neurons=Options.Neurons(1);    % Network Architecture
lambdaMin=Options.lambdaMin;   % minimal value of  forgeting factor
mu=Options.mu;                 % sensitivity factor
gamma=Options.gamma;           % Discount factor
C=Options.C;                   % regularization parameter
% store the first value of forgetting factor
lambdas(1,1)=lambdaMin;
% Start training 
start_time_train=cputime;
% initialize the model 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% load initial mini-batches                                           
P = Trinputs{1}; 
T = Trtargets{1};
% generate input weights
input_weights=rand(Neurons,size(P,2));         
% calculating the temporal hidden layer 
% formula (1) 
tempH=input_weights*P';                 
% Activation function
switch lower(activF)
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
    case {'sig'}
        %%%%%%%% sigmoid function
        H = SigActFun(tempH);
%%%%%% More activation functions can be added here                
end 
% Save the new features representations  
Hn{1}=H; 
% Calculate beta
% formula (2) 
B=pinv(H') * T; 
% Calculate the covariance matrix
% formula (8)
M = pinv((1/C)+H * H');
% end  of initialization of the model
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% start sequential Learning
% initialize counters
% formula (5)
E{1}=T-scaledata((H'*B),min(T),max(T)); % initial TD error
e(1)=sqrt(mean(E{1}));                   % initial RMSE of TD
c=0;                                    % counter of accepted mini_batches
if numel(Trinputs)>1%
    
for t=2:numel(Trinputs)
% Load new mini-batches 
Pnew = Trinputs{t};
Tnew = Trtargets{t};
% Temporal hidden layer

Hnew_temp=input_weights*Pnew';

%%%%%Activation function%%%
 switch lower(activF)
    case {'radbas'}
        %%%%%%%% Radial basis function
        Hn{t} = radbas(Hnew_temp);
    case {'sig'}
        %%%%%%%% sigmoid function
        Hn{t} = SigActFun(Hnew_temp);
        %%%%%%%% More activation functions can be added here                
 end
 
% calculate TD error
% formula (10)
E{t}=Tnew-scaledata(((Hn{t}-gamma*Hn{t-1})' * B),min(Tnew),max(Tnew));
% calculate RMSE of TD
e(t)=sqrt(mean((E{t}).^2));% 
% USS (updated sellection strategy )
% condition in formula (9)
lambda=lambdaMin; % initialize
if sqrt(mean((E{t}).^2))>sqrt(mean((E{t-1}).^2))
c=c+1;
% Store index of accepted mini_batches 
Index_Accepted(1,c)=t; % accepted mini_batches
% Update lambda 
% formula (13)
lambda=lambdaMin+(1-lambdaMin)*exp(-mu*sqrt(mean((E{t}).^2)));
% formula (13)
% Boundary constraints adjustement for lambda (forgetting factor)
if lambda<=lambdaMin
lambda=lambdaMin;
elseif lambda>=1
lambda=1;    
end
% Update beta

% Gain matrix
% formula (12)
K =(M * Hn{t}) *((lambda+eye(size(Pnew,1))+(Hn{t}-gamma*Hn{t-1})' * M * Hn{t})^(-1));
% Covariance matrix
% formula (11)
M = (1/lambda)*(M -( K * (Hn{t}-gamma*Hn{t-1})' * M));
% Output weights beta
% formula (9)
B = B + M * Hn{t} * (E{t});
end
% Store forgetting factors
lambdas(1,t)=lambda;
end
end

end_time_train=cputime;
Tr_Time=end_time_train-start_time_train;
% End of the sequential phase
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Training targets
for i=1:numel(Trinputs)
% calculate the hidden layer
Htrain=(input_weights*Trinputs{i}');
%%%%%Activation function%%%
switch lower(activF)
    case {'radbas'}
        %%%%%%%% Radial basis function
        Htrain = radbas(Htrain);
    case {'sig'}
        %%%%%%%% Sigmoid function
        Htrain = SigActFun(Htrain);
      %%%%%%%% More activation functions can be added here                
end
% Estimate the targets of training
Y_hat{i}=scaledata(Htrain'* B,min(Trtargets{i}),max(Trtargets{i}));
end
% Training RMSE
[~,~,~,error]=Score(Y_hat,Trtargets);
tr_acc=mean(error);
start_testing_time=cputime;
% Testing targets
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for i=1:numel(Tsinputs)
% calculate the hidden layer
Htest=(input_weights*Tsinputs{i});
%%%%%Activation function%%%
switch lower(activF)
     case {'radbas'}
        %%%%%%%% Radial basis function
        Htest = radbas(Htest);
    case {'sig'}
        %%%%%%%% Sigmoid function
        Htest = SigActFun(Htest);
      %%%%%%%% More activation functions can be added here                
end
% estimate the targets of testing
Yts_hat{i}=scaledata(Htest'* B,min(Tstargets{i}),max(Tstargets{i}));
end
Ts_Time=cputime-start_testing_time;
% testing  accuracy (Model evaluation)
[SCORE,S,d,er]=Score(Yts_hat,Tstargets);
ts_acc=mean(er);
%%%%%%%%%%%%%%%%% Save results

 net.Tr_Time=Tr_Time;      % training time
 net.Ts_Time=Ts_Time;      % testing time
% net.Y=Trtargets;          % original output of training set
% net.Yts=Tstargets;        % original output for testing set
% net.Y_hat=Y_hat;          % estimated output for training set
% net.Yts_hat=Yts_hat;      % estimated output for testing set
  net.S_value=SCORE;        % the score value
% net.S_vector=S;           % the score vector (for each sample)
% net.d_vecteur=d;          % the differance vector (for each sample)
% net.R_vector=er;          % the RMSE vector (for each sample)
  net.tr_acc=tr_acc;        % RMSE of training 
  net.ts_acc=ts_acc;        % RMSE of testing 
% net.Index=Index_Accepted; % indexes of accepted mini batches
% net.lambdas=lambdas;      % forgetting factors
% net.R_TrM=e;              % RMSE of TD of each training mini-batch
                
end
