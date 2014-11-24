clear all

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

X =[word_train bigram_train];
Y = price_train;
%%
disp('PCA...')
% X_pca = spca(X, 100);
load X_pca
X_pca = X_pca(:,1:250);
disp('done!')
X_pca_city = full([city_train X_pca]);
%%

n_bins = 10;
bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
Y_class = zeros(size(Y));

for i = 2:(n_bins+1)
    Y_class(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
end


n_part = 10;
xval_part = make_xval_partition(numel(Y), n_part);

err = zeros(n_part,1);

RegTreeTemp = templateTree('Surrogate','On');

for i = 1:n_part
    fprintf('Xval partition # %d \n', i);
    
    X_train = X_pca_city(xval_part ~= i, :);
    Y_train = Y_class(xval_part ~= i,:);
    X_test = X_pca_city(xval_part == i, :);
    Y_test = Y_class(xval_part == i,:);
    
    dt = dt_train_multi(X_train, Y_train, 10);
    Yhat_test = zeros(size(Y_test));
    
    for j = 1:size(X_test,1)
        [~, Yhat_test(j)] = max(dt_value(dt, X_test(j,:)));
    end
    err(i) = mean(Yhat_test ~= Y_test);
end

fprintf('Xval Err: %f \n', mean(err));
