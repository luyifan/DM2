function [X_test, y_test] = maketestdata( w_f , nTest , noisy )

P=size(w_f, 1);

X_test=[] ;
y_test=[] ;
for iter=1:nTest
    while true
        oneX = rand ( P , 1 ) ;
        oneX(1)= 1;
        oneY = w_f*oneX' ;
        if ( oneY > 0 )
            yy = 1 ;
        
        else
            yy = 0 ;
        end
        if abs( oneY ) > 1E-2
            break
        end
    end
    X_test = [ X_test , oneX];
    y_test = [ y_test , yy ]; 
end

X_test = X_test ( 2:end , : );

N=nTest;
if nargin == 3
    if noisy == 'noisy'
        idx = randsample(N,N/10);
        y_test(idx) = -y_test(idx);
    end
end

end