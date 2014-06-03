function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
iter = 0 ;
P = size ( X , 1 ) ;
N = size ( X , 2 ) ;
w = rand( P+1 , 1 );
X = [ ones(1,N) ; X ];
eta = 1;
theta = 0.1;
maxiter = 1000 ;
while true
    iter=iter+1
    Yk=find(sign(w'*X.*y)<0);
    update = eta.*(X(:,Yk)*y(:,Yk)');
    w=w+update;
    if( ( norm(update,2) < theta )|( iter > maxiter  ) )
        break;
    end
end


end
