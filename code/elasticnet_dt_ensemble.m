%% control variables
full_prediction = 1;
cross_val = 0;

%% CVGLMNET for different cities, then use a DT forest to fit the residual
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

x_train = [city_train word_train bigram_train];
y_train = price_train;
x_test = [city_test word_test bigram_test];

% if a pool is open, close it
try
    matlabpool('close');
catch err
end

worker_pool = parpool();

if cross_val == 1
    mse = crossval('mse', x_train, y_train,'Predfun', ...
        @elasticnet_dt_ensemble_pred_fun, 'kfold', 2)
end

if full_prediction == 1
    yfit = elasticnet_dt_ensemble_pred_fun(x_train, y_train, x_test);
    dlmwrite('submit.txt', yfit)
end


delete(worker_pool);
