function mse = xval_test(X, Y, predfun, n_parts)

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

X =[city_train word_train bigram_train];
Y = price_train;

initialize_additional_features;

%% Run algorithm

mse = crossval('mse',X,Y,'Predfun',predfun, 'kfold', n_parts);

end