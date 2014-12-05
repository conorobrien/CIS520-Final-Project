addpath('../glmnet_matlab/')
load ../../data/city_train.mat
load ../../data/city_test.mat
load ../../data/word_train.mat
load ../../data/word_test.mat
load ../../data/bigram_train.mat
load ../../data/bigram_test.mat
load ../../data/price_train.mat

% if a pool is open, close it
try
    matlabpool('close');
catch err
end
worker_pool = parpool();

x_train = [word_train bigram_train];
y_train = price_train;
%x_test = [city_test word_test bigram_test];
opts = statset('display', 'iter', 'UseParallel', true);
inmodel = sequentialfs(@sequentialfs_fun, full(x_train), y_train, 'options', opts);

delete(worker_pool);
