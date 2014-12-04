clear;
load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

addpath glmnet_matlab liblinear scratch libsvm

X =[city_train word_train bigram_train];
Y = price_train;

xval_part = make_xval_partition(length(Y_train), 10);

err = [];

for i = 1:1

	X_train = X(xval_part ~= i, :);
	Y_train = Y(xval_part ~= i, :);
	X_test = X(xval_part == i, :);
	Y_test = Y(xval_part == i, :);
	%% Run algorithm
	% Example by lazy TAs

    Y_hat = elasticnet_dt_ensemble_pred_fun(X_train,Y_train,X_test);
	err(i) = rmse(Y_hat, Y_test)

end


%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');