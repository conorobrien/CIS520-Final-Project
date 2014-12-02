clear variables

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

X =[word_train bigram_train];
Y = price_train;
%%
disp('PCA...')
X_pca = spca(X, 100);
disp('done!')
X_pca_city = [city_train X_pca];

%%
disp('XVAL ...')
mse = crossval('mse',X_pca_city,Y,'Predfun',@linear_pred_test, 'kfold', 1)
disp('done!')
