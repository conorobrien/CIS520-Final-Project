clearvars

addpath('liblinear/matlab')

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

X =[word_train bigram_train];
X_pca = spca(X,100);
X_pca_city = [city_train X_pca];
Y = price_train;

%%
n_bins = 5;
bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
Y_class = zeros(size(Y));

for i = 2:(n_bins+1)
    Y_class(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
end

%%
n_part = 10;
xval_part = make_xval_partition(numel(Y), n_part);
err = zeros(n_part,1);

tic
for i = 1:n_part
    fprintf('Xval partition # %d \n', i);
    
    X_train = X_pca_city(xval_part ~= i, :);
    Y_train = Y_class(xval_part ~= i, :);
    X_test = X_pca_city(xval_part == i, :);
    Y_test = Y_class(xval_part == i, :);
    
    model = train(Y_train,X_train,[,'-t 3']); %#ok<NBRAK,NOCOM>
    Yhat = predict(Y_test, X_test, model);
    err(i) = mean(Yhat ~= Y_test);
end
toc
fprintf('Xval Err: %f \n', mean(err));

% 20 bins, full sparse X, err = 