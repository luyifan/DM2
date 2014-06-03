%% Ridge Regression
load('digit_train', 'X', 'y');
load('digit_test', 'X_test', 'y_test');
% Do feature normalization
% ...

show_digit(X);
X=X - repmat ( mean ( X , 2 ) , 1 , size ( X , 2 ) ) ;
stdX=repmat( std( X , 0 , 2 ) , 1 , size ( X , 2 ));
X=X./stdX;
figure
show_digit(X);


w=ridge ( X , y , 0 ) ;
fprintf('Without regularization  sum ( w^2 ) is %f\n', sum( w.*w ) );  
error = errornum( w , X , y );
error = error/size ( y , 2 ) ;
fprintf('Without regularization  E_train is %f\n', error);

X_test=X_test - repmat ( mean ( X_test , 2 ) , 1 , size ( X_test , 2 ) ) ;
stdX=repmat( std( X_test , 0 , 2 ) , 1 , size ( X_test , 2 ));
X_test=X_test./stdX;
% Compute test error
error = errornum( w , X_test , y_test  );
error = error/size ( y_test , 2 ) ;
fprintf('Without regularization E_test is %f\n', error);


% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda=0;
Min_E=1e20;
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_n = [ X(:,1:j-1) , X(:,j+1:end)]; 
        y_n = [ y(:,1:j-1) , y(:,j+1:end)]; % take point j out of X
        w = ridge(X_n, y_n, lambdas(i));
        E_val = E_val + (sign(w'*[1;X(:,j)]*y(:,j))>0);
    end
    % Update lambda according validation error
    if ( E_val < Min_E )
       lambda=lambdas(i);
       Min_E = E_val ;
    end
end

% Compute training error

fprintf('Lambda we choose is %f\n', lambda );
w = ridge ( X , y , lambda );
error = errornum( w , X , y  );
error = error/size ( y , 2 ) ;
fprintf('Ridge Regression  E_train is %f\n', error);

% Do feature normalization
X_test=X_test - repmat ( mean ( X_test , 2 ) , 1 , size ( X_test , 2 ) ) ;
stdX=repmat( std( X_test , 0 , 2 ) , 1 , size ( X_test , 2 ));

X_test=X_test./stdX;

% Compute test error
error = errornum( w , X_test , y_test );
error = error/size ( y_test , 2 ) ;
fprintf('Ridge Regression E_test is %f\n', error);


%% Logistic

%% SVM with slack variable
