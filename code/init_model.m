function model = init_model()

%% Load glmnet regression models
tmp = load('glmnet_fit.mat');
model.cvglmnet_fit = tmp.cvglmnet_fit;

%% Load trees on residual
tmp = load('tree_fit.mat');
model.tree_fit = tmp.tree_fit;
