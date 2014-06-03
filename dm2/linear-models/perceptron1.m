function [w, iter] = perceptron1(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
	iter = 1;
	eta = 1;
	theta = 0.1;
    maxiter = 10000;
	P = size(X,1);
	N = size(X,2);
	one = ones(1,N);
	XX =[one;X];
	w=rand(P+1,1);
	while (iter<maxiter)
		deta = eta.*(XX(:,find(sign(w'*XX)~=y))*y(:,find(sign(w'*XX)~=y))');
		%{
		tmp_X = XX;
		tmp_y = y;
		tmp_X(:,find(sign(w'*XX)==y)) = [];
		tmp_y(:,find(sign(w'*XX)==y)) = [];
		deta = yita.*(tmp_X*tmp_y');
		%}
		w = w + deta;
		if(norm(deta,2) < theta)
			break;
		end
		iter = iter +1;
	end
end
