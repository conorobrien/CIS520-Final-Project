addpath('glmnet_matlab')
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

x_train =[bigram_train];
%x_train = spca(x_train, 30); 
y_train = price_train;

small_clus_idx = find(abs(x_train(:, 3)) > 4);
big_clus_idx = find(abs(x_train(:, 3)) <= 4);

y_hat = zeros(size(y_train));

x_big = x_train(big_clus_idx);
y_big = y_train(big_clus_idx);
m_big = cvglmnet(x_big, y_big, 'gaussian');
y_hat(big_clus_idx) = cvglmnetPredict(m_big, x_big);

x_small = x_train(small_clus_idx);
y_small = y_train(small_clus_idx);
m_small = cvglmnet(x_small, y_small, 'gaussian');
y_hat(small_clus_idx) = cvglmnetPredict(m_small, x_small);

mse = sqrt((sum(((y_hat - y_train) .^ 2))/numel(y_train)))
