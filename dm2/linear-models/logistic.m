function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
N=size ( X , 2 ) ;
P=size ( X , 1 ) ;
X=[ones(1,N);X];
iter=0;
maxiter=1000;
esp=0.001;
alpha=0.5;

w=zeros(P+1,1);

while true
  iter=iter+1;
  g=-w'*X;
  h=1./(1+exp(g));
  update=alpha*X *(y-h)';
  loss = sum ( abs( update ) );
  w=w+update;
  if ( loss < esp )|( iter > maxiter )
      break
  end
end
disp ( iter )

end
