clearvars

addpath('liblinear/matlab')
addpath('libsvm');

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat
% 
% word_train = word_train(:,sum(word_train)>50);
% bigram_train = bigram_train(:, sum(bigram_train)>75);
% 
X =[word_train bigram_train];
% X_pca = spca(X,200);
load X_pca
X_pca = X_pca(:,1:400);
% 
X_pca = bsxfun(@minus, X_pca, mean(X_pca));
X_pca = bsxfun(@rdivide, X_pca, max(abs(X_pca)));

X_pca_city = [city_train X_pca];
X_city = [city_train X];
Y = price_train;

%
n_bins = 3;
bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
Y_class = zeros(size(Y));

for i = 2:(n_bins+1)
    Y_class(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
end

% This code sorts the prices into evenly-sized bins, is worse than
% even-width bins
% n_bins = 10;
% bin_length = floor(length(Y)/n_bins);
% Y_class = zeros(size(Y));
% [~, i_sorted] = sort(Y);
% 
% for i = 0:(n_bins-2)
%     index = i_sorted((i*bin_length +1):((i+1)*bin_length));
%     Y_class(index) = i+1;
% end
% index = i_sorted((end-bin_length):end);
% Y_class(index) = n_bins;

%%
n_part = 10;
xval_part = make_xval_partition(numel(Y), n_part);
err = zeros(n_part,1);

tic
for i = 1:1
    fprintf('Xval partition # %d \n', i);
    
    X_train = X_pca_city(xval_part ~= i, :);
    Y_train = Y_class(xval_part ~= i, :);
    X_test = X_pca_city(xval_part == i, :);
    Y_test = Y_class(xval_part == i, :);
    
    model = svmtrain(Y_train,X_train,[,'-t 2 -q']); %#ok<NBRAK,NOCOM>
    Yhat = svmpredict(Y_test, X_test, model);
    err(i) = mean(Yhat ~= Y_test);
end
toc
fprintf('Xval Err: %f \n', max(err));

% 20 bins, full sparse X, err = 