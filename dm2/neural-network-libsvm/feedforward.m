function p = feedforward(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%
%   Input:  Theta1 -- weights between input-hidden layers, 401x25 matrix
%           Theta2 -- weights between hidden-output layers, 26x10 matrix
%                X -- test set, 400xP matrix, P is size of testing set
%
%   Output: p -- predicted labels, 1xP row vector

% Note:
% The matrix X contains the examples in columns.
% The matrices Theta1 and Theta2 contain the parameters for each unit in
% column. Specifically, the first column of Theta1 corresponds to the first
% hidden unit in the second layer.

% YOUR CODE HERE
XX=[ones(1,size(X,2));X];
Y=1./(1+exp(-Theta1'*XX));
YY=[ones(1,size(Y,2));Y];
Z=1./(1+exp(-Theta2'*YY));
[val,p]=max(Z);
end
