function model = init_model()

addpath('glmnet_matlab');
%% Load glmnet regression models
tmp = load('model_alt.mat');

model.cvglmnet_fit = tmp.cvglmnet_fit;

% convert beta back to a double (might not be necessary, but some fcns prefer
% double)
for i = 1:7
    model.cvglmnet_fit{i}.glmnet_fit.beta = double(model.cvglmnet_fit{i}.glmnet_fit.beta);
end

%% Load trees on residual
model.tree_fit = tmp.tree_fit;

%% Load Feature Index
model.idx = logical([zeros(1,7) tmp.idx]);
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

