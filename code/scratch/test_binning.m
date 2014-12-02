clearvars

addpath('liblinear/matlab')
addpath('libsvm');

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat
% 
city = 5;
word_train = word_train(:,sum(word_train)>100);
bigram_train = bigram_train(:, sum(bigram_train)>150);
word_train = word_train(city_train(:,city) == 1,:);
bigram_train = bigram_train(city_train(:,city) == 1,:);

disp('PCA...')
X = full([spca(word_train,200) spca(bigram_train,200)]);
Y = full(price_train);
disp('Done')
% X = X(X(:,5) == 1, :);
Y = Y(city_train(:,city) == 1, :);

%%
n_bins = 2;
bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
Y_bins_real = zeros(size(Y));

for i = 2:(n_bins+1)
    Y_bins_real(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
end    
    
    
%%
n_part = 5;
xval_part = make_xval_partition(numel(Y), n_part);
reg_err = zeros(n_part,1);
bin_err = reg_err;

tic
for i = 1:1
    fprintf('Xval partition # %d \n', i);
    
    X_train = X(xval_part ~= i, :);
    Y_train = Y(xval_part ~= i, :);

    X_test = X(xval_part == i, :);
    Y_test = Y(xval_part == i, :);
    
%     [Yhat_bins, Y_bins] = predict_bins(X_train, Y_train, X_test, n_bins);
%     price_pred = elastic_net_bins(X_train, Y_train, X_test, Y_bins, Yhat_bins);
    [Yhat_bins,Yhat_bins_score, Y_bins] = predict_bins_score(X_train, Y_train, X_test, n_bins);
    price_pred = elastic_net_bin_scores(X_train, Y_train, X_test, Y_bins, Yhat_bins_score);
    bin_err = mean(Y_bins_real(xval_part == i) ~= Yhat_bins);
    reg_err = rmse(price_train(xval_part == i), price_pred);
end
toc

fprintf('Binning Xval Err: %f \n', max(bin_err));
fprintf('Regression Xval Err: %f \n', max(reg_err));

% 20 bins, full sparse X, err = 