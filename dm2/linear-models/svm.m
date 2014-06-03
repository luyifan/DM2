function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
options = optimset;    
options.LargeScale = 'off';
options.Display = 'off';
C=1;
H=(y'*y).*(X'*X);
n=length(y);
f=-ones(n,1);
A=[];
b=[];
Aeq=y;
beq=0;
lb=zeros(n,1);
ub=C*ones(n,1);
a0=zeros(n,1);
[a,fval,eXitflag,output,lambda]= quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
size(a)
ww=X*(a.*y');
w=[1,2,3];
epsilon=1e-8;
i_sv=find(abs(a)>epsilon);
num=size(i_sv,1);
%b=mean(y(i_sv) - ww'*X(:,i_sv));
b=mean(y(i_sv)-ww'*X(:,i_sv));

w = [ b ; ww  ] ;
end
