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
X_pca = X_pca(:,1:150);
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
% 
% pool = parpool;
% options = statset;
% options.UseParallel = true;

err = zeros(n_part,1);
tic
for i = 1:1
    fprintf('Xval partition # %d \n', i);
    
    X_train = X_pca_city(xval_part ~= i, :);
    Y_train = Y_class(xval_part ~= i,:);
    X_test = X_pca_city(xval_part == i, :);
    Y_test = Y_class(xval_part == i,:);
    
%     trees = TreeBagger(200,X_train, Y_train,'options', options);
    trees = fitensemble(X_train,Y_train,'AdaBoostM2',400,'Tree');

    Yhat_test = trees.predict(X_test);

%     Yhat_test_num = zeros(size(Yhat_test));
%     
%     for j = 1:numel(Yhat_test)
%         Yhat_test_num(j) = str2double(Yhat_test{j});
%     end
    
    err(i) = mean(Yhat_test ~= Y_test);
    fprintf('MSE: %f \n', err(i));
end
toc
fprintf('Xval Err: %f \n', mean(err));

delete(pool)