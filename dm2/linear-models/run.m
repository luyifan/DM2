% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron

nRep = 100; % number of replicates
nTrain = 100; % number of training data

TotalError_train=0;
TotalError_test=0;
TotalIter = 0 ;
for i = 1:nRep
    i
    [X, y, w_f] = mkdata(nTrain);
    [w_g, iter] = perceptron1(X, y);
    
    % Compute training, testing error
    error = errornum ( w_g ,  X , y );
    TotalError_train=TotalError_train+error;
    TotalIter=TotalIter+iter;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdata ( w_f , nTest ) ;
     
    error=errornum ( w_g , X_test , y_test );
    TotalError_test=TotalError_test+error;
   
end

E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTrain*10*nRep);
avgIter=TotalIter/nRep;
fprintf('Preceptron E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Preceptron Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron');


%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y);
error = errornum ( w_g ,  X , y );
error =error/nTrain;
disp(['The mean error rate in noisy : ' num2str(error*100, '%.2f%%')]);
plotdata(X, y, w_f, w_g, 'Pecertron');



%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

TotalError_train=0;
TotalError_test=0;

for i = 1:nRep
    i
    [X, y, w_f] = mkdata(nTrain);
    w_g = linear_regression(X, y);
    % Compute training, testing error
    
    error = errornum ( w_g ,  X , y );
    TotalError_train=TotalError_train+error;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdata ( w_f , nTest ) ;
     
    error=errornum ( w_g , X_test , y_test );
    TotalError_test=TotalError_test+error;
   
end
E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTrain*10*nRep);
fprintf('Linear Regression E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');


%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

TotalError_train=0;
TotalError_test=0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = linear_regression(X, y);
    % Compute training, testing error
    error = errornum ( w_g ,  X , y );
    TotalError_train=TotalError_train+error;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdata ( w_f , nTest , 'noisy') ;
     
    error=errornum ( w_g , X_test , y_test );
    TotalError_test=TotalError_test+error;
end
E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTrain*10*nRep);

fprintf('Linear Regression: noisy E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');



%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');

w = linear_regression(X, y);
error = errornum ( w ,  X , y );
E_train=error/size(X,2);

error = errornum ( w , X_test , y_test );
E_test=error/size(X_test,2);
%Compute training, testing error
fprintf('Linear Regression: poly_fit E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
X_t = X; % CHANGE THIS LINE TO DO TRANSFORMATION
X_t=[X_t;ones(3,size(X_t,2))];
X_t(3,:)=X_t(1,:).*X_t(2,:);
X_t(4,:)=X_t(1,:).*X_t(1,:);
X_t(5,:)=X_t(2,:).*X_t(2,:);
X_test_t = X_test; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t=[X_test_t;ones(3,size(X_test_t,2))];
X_test_t(3,:)=X_test_t(1,:).*X_test_t(2,:);
X_test_t(4,:)=X_test_t(1,:).*X_test_t(1,:);
X_test_t(5,:)=X_test_t(2,:).*X_test_t(2,:);
w=linear_regression(X_t, y);
% Compute training, testing error
error = errornum ( w ,  X_t , y );
E_train=error/size(X,2);

error = errornum ( w , X_test_t , y_test );
E_test=error/size(X_test,2);

fprintf('poly_fit with transform E_train is %f, E_test is %f.\n', E_train, E_test);



%% Part6: Logistic Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
TotalError_train=0;
TotalError_test=0;
for i = 1:nRep
    i
    [X, y, w_f] = mkdata(nTrain);
    w_g = logistic(X, y);
    
    % Compute training, testing error
    error = errornum ( w_g ,  X , y , 'logistic');
    TotalError_train=TotalError_train+error;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdataLogistic ( w_f , nTest ) ;
     
    error=errornum ( w_g , X_test , y_test ,'logistic');
    TotalError_test=TotalError_test+error;
end

E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTrain*10*nRep);
fprintf('Logistic RegressionE_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 1000; % number of replicates
nRep=5;
nTrain = 100; % number of training data
nTest = 10000; % number of training data
TotalError_train=0;
TotalError_test=0;
for i = 1:nRep
    i
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = logistic(X, y);
    
    % Compute training, testing error
    error = errornum ( w_g ,  X , y , 'logistic');
    TotalError_train=TotalError_train+error;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdataLogistic ( w_f , nTest , 'noisy') ;
     
    error=errornum ( w_g , X_test , y_test ,'logistic');
    TotalError_test=TotalError_test+error;
end

E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTest*nRep);
fprintf('Logistic Regression: noisy E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');


%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 30; % number of training data
TotalError_train=0;
TotalError_test=0;
TotalNum_sc=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    [w_g, num_sc] = svm(X, y);
    plotdata(X, y, w_f, w_g, 'SVM');
    % Compute training, testing error
    error = errornum ( w_g ,  X , y );
    TotalError_train=TotalError_train+error;
    
    nTest = nTrain * 10 ;
    [X_test,y_test ] = maketestdata ( w_f , nTest ) ;
   
    error=errornum ( w_g , X_test , y_test );
    TotalError_test=TotalError_test+error;
    % Sum up number of support vectors
    TotalNum_sc = TotalNum_sc + num_sc ;
end
E_train = TotalError_train/(nTrain*nRep);
E_test = TotalError_test/(nTest*nRep);
avg_num_sc= TotalNum_sc/nRep;

fprintf('support vectors E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of support vectors is %f.\n', avg_num_sc );
