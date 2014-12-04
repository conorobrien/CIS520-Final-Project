function model = init_model()

addpath('glmnet_matlab');
%% Load glmnet regression models
tmp = load('cvglmnet_fit_1se_new.mat');

model.cvglmnet_fit = tmp.cvglmnet_fit;

% convert beta back to a double (might not be necessary, but some fcns prefer
% double)
for i = 1:7
    model.cvglmnet_fit{i}.glmnet_fit.beta = double(model.cvglmnet_fit{i}.glmnet_fit.beta);
end

%% Load trees on residual
tmp = load('tree_fit_1se_deep.mat');
model.tree_fit = tmp.tree_fit;
