clear all

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

pool = parpool
options = statset;
options.UseParallel = true;
%%
n_part = 10;
xval_part = make_xval_partition(numel(Y), n_part);

rmse_ens = zeros(n_part,1);
rmse_tree = rmse_ens;
rmse_lin = rmse_ens;

RegTreeTemp = templateTree('Surrogate','On');

for i = 1:1
    fprintf('Xval partition # %d \n', i);
    Xtrain = X_pca_city(xval_part ~= i, :);
    Ytrain = Y(xval_part ~= i,:);
    Xtest = X_pca_city(xval_part == i, :);
    Ytest = Y(xval_part == i,:);
    
%     trees = fitrtree(full(Xtrain), Ytrain);
    trees = TreeBagger(2000,X_pca_city,Y,'method','regression', 'options', options);
%     trees = fitensemble(full(Xtrain),Ytrain,'Bag', 1500,RegTreeTemp, 'type', 'regression');
    Yhat_tree = trees.predict(full(Xtest));
    
    Yhat_lin = linear_pred_test(Xtrain, Ytrain, Xtest);
    
    Yhat = 0.5*Yhat_tree + 0.5*Yhat_lin;
    rmse_tree(i) = sqrt(mean((Ytest - Yhat_tree).^2));
    rmse_lin(i) = sqrt(mean((Ytest - Yhat_lin).^2));
    rmse_ens(i) = sqrt(mean((Ytest - Yhat).^2));
    fprintf('RMSE Tree Reg.: %f \n',rmse_tree(i));
    fprintf('RMSE Linear Reg.: %f \n',rmse_lin(i));
    fprintf('RMSE Ensemble: %f \n',rmse_ens(i));

end

% For ensemble learner, it's not great and it takes forever
% ens = TreeBagger(200,X_pca_city,Y,'method','regression')
