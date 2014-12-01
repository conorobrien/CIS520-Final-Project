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

worker_pool = parpool();
mse = crossval('mse', x_train, y_train,'Predfun', ...
    @elasticnet_dt_ensemble_pred_fun, 'kfold', 2)
delete(worker_pool);
