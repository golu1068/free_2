function H = SigActFun(V);

%%%%%%%% Feedforward neural network using sigmoidal activation function

H = 1./(1+exp(-V));