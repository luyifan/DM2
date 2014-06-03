load('digit_data', 'X', 'y');
load('weights', 'Theta1', 'Theta2');
addpath('./libsvm-3.18/matlab');

p = feedforward(Theta1, Theta2, X);
fprintf('Error rate for NN is %f.\n', length(find(p ~= y))/length(p));

train_X = X(:, 1:2500);
train_y = y(1:2500);
test_X = X(:, 2501:end);
test_y = y(2501:end);

% YOUR CODE HERE

% Trainning and testing using one-vs-all with LIBLINEAR=
predict_pro=ones(2500,10);
for i=1:10
    i
	train_label =ones(1,2500);
	train_label(find(train_y==i))=1;
	train_label(find(train_y~=i))=-1;
	model = svmtrain( train_label',train_X','-c 1 -t 1 -b 1');
	[label, accuracy, prob] = svmpredict(test_y',test_X', model,'-b 1');
	predict_pro(:,i) = prob(:,1);
end
    
[val,label]=max(predict_pro,[],2);
error=size(find(label'~=test_y) , 2)/2500;
fprintf('Error rate for SVM is %f.\n', error);
