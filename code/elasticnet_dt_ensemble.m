%% control variables
full_prediction = 0;
cross_val = 1;

addpath glmnet_matlab scratch libsvm
%% CVGLMNET for different cities, then use a DT forest to fit the residual
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

% load loading1000.mat
% x_train_pca = [word_train bigram_train] * V1;
% x_test_pca = [word_test bigram_test] * V1;
% 
% x_train = [city_train x_train_pxa];
% y_train = price_train;
% x_test = [city_test x_test_pca];

x_train = [city_train word_train bigram_train];
y_train = price_train;
x_test = [city_test word_test bigram_test];

%assign pool
worker_pool = gcp('nocreate');

if cross_val == 1
   % mse = crossval('mse', x_train, y_train,'Predfun', ...
        % @elasticnet_dt_ensemble_pred_fun, 'kfold', 10);
   xval_part = make_xval_partition(length(y_train), 10);
   
   err = [0 0];
   for i = 1:2
       xtr = x_train(xval_part ~= 1,:);
       xte = x_train(xval_part == 1,:);
       ytr = y_train(xval_part ~= 1,:);
       yte = y_train(xval_part == 1,:);

       yfit = elasticnet_dt_ensemble_pred_fun(xtr, ytr, xte);

       err(i) = rmse(yfit, yte)
   end
   
end

if full_prediction == 1
    yfit = elasticnet_dt_ensemble_pred_fun(x_train, y_train, x_test);
    dlmwrite('submit.txt', yfit)
end

beep
beep

% delete(worker_pool);
