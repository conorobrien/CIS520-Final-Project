function model = init_model()

addpath('glmnet_matlab');
%% Load glmnet regression models
tmp = load('cvglmnet_fit_alt.mat');

model.cvglmnet_fit = tmp.cvglmnet_fit;

% convert beta back to a double (might not be necessary, but some fcns prefer
% double)
for i = 1:7
    model.cvglmnet_fit{i}.glmnet_fit.beta = double(model.cvglmnet_fit{i}.glmnet_fit.beta);
end

%% Load trees on residual
tmp = load('tree_fit_alt.mat');
model.tree_fit = tmp.tree_fit;
model.tree_fit2 = tmp.free_fit2;

%% Calculate PCA coefficients
% load ../data/city_train.mat
% load ../data/city_test.mat
% load ../data/word_train.mat
% load ../data/word_test.mat
% load ../data/bigram_train.mat
% load ../data/bigram_test.mat
% 
% X = [city_train word_train bigram_train;
%      city_test word_test bigram_test]; 
% [~,~,model.pca_coeff] = spca(X,100);

