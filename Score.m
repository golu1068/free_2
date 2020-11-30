function[SCORE,S,d,er]=Score(YPred,Y)
% use : formula (17) and (18)
%
% Scoring Function
% YPred    : estimated Targets
% Y        : desired Targets
% SCORE    : Score
% S        : Score value for each sample 
% d        : error value for each sample 
% er       : RMSE error value for each sample 
%
%
result=[];rul1=[];
for i = 1:numel(YPred)
    R(i)=Y{i}(end);
    YPredLast(i) = YPred{i}(end);
    result=[ result; YPredLast(i)];
    rul1=[rul1;R(i)];
end
earlyPenalty = 13;
latePenalty  = 10;
d = result - rul1;
S = zeros(size(d));
%%% LATE prediction, pred > trueRUL, pred - trueRUL > 0

f = find(d >= 0);
FNR = length(f) / length(rul1); % false negative rate
S(f) = exp(d(f)/latePenalty)-1;
%%% EARLY prediction,pred < trueRUL, pred - trueRUL < 0

f = find(d < 0);
FPR  =  length(f) / length(rul1);% false positive rate
S(f) = exp(-d(f)/earlyPenalty)-1;
SCORE=sum(S);
er=sqrt(d.^2);
end