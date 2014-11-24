clearvars

addpath('liblinear/matlab')

% load ../data/city_train.mat
% load ../data/word_train.mat
% load ../data/bigram_train.mat
% load ../data/price_train.mat
% 
% X =[word_train bigram_train];
% X_pca = spca(X,250);
load X_pca
X_pca = X_pca(:,1:250);
X_pca_city = [city_train X_pca];
Y = price_train;

%%
n_part = 10;
xval_part = make_xval_partition(numel(Y), n_part);
err = zeros(n_part,1);

tic
for i = 1:n_part
    fprintf('Xval partition # %d \n', i);
    
    X_train = X_pca_city(xval_part ~= i, :);
    Y_train = Y(xval_part ~= i, :);
    X_test = X_pca_city(xval_part == i, :);
    Y_test = Y(xval_part == i, :);
    
    model = train(Y_train,X_train,[,'-s 11']); %#ok<NBRAK,NOCOM>
    Yhat = predict(Y_test, X_test, model);
    err(i) = rmse(Yhat, Y_test);
end
toc
fprintf('Xval Err: %f \n', mean(err));

