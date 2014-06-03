function [ error ] = errornum( w , X , y )

XX = [ ones(1,size(X,2)); X ] ;
N=size ( y , 2 );
error=size( find(sign((w'*XX+ones(1,N)).*y)<0) , 2)  ;      

end
