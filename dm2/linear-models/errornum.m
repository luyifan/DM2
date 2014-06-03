function [ error ] = errornum( w , X , y , logistic )

XX = [ ones(1,size(X,2)); X ] ;
error=size( find(sign(w'*XX.*y)<0) , 2)  ;      
if nargin == 4
    if logistic == 'logistic'
        g=-w'*XX;
        error=size(find(sign(1./(1+exp(g)).*y)<0) , 2);
    end
end
end