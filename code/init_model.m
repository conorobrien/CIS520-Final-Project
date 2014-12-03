function model = init_model()

addpath('glmnet_matlab');
%% Load glmnet regression models
tmp = load('cvglmnet_fit_1se.mat');

model.cvglmnet_fit = tmp.cvglmnet_fit;

% convert beta back to a double (might not be necessary, but some fcns prefer
% double)
% Everything's actually faster when leaving it as a single precision
% for i = 1:7
%     model.cvglmnet_fit{i}.glmnet_fit.beta = double(model.cvglmnet_fit{i}.glmnet_fit.beta);
% end

% Convert everything else to single, this kindof slows everything down
% though
% for i = 1:7
%     model.cvglmnet_fit{i}.glmnet_fit.a0 = single(model.cvglmnet_fit{i}.glmnet_fit.a0);
%     model.cvglmnet_fit{i}.glmnet_fit.dev = single(model.cvglmnet_fit{i}.glmnet_fit.dev);
%     model.cvglmnet_fit{i}.glmnet_fit.nulldev = single(model.cvglmnet_fit{i}.glmnet_fit.nulldev);
%     model.cvglmnet_fit{i}.glmnet_fit.df = single(model.cvglmnet_fit{i}.glmnet_fit.df);
%     model.cvglmnet_fit{i}.glmnet_fit.lambda = single(model.cvglmnet_fit{i}.glmnet_fit.lambda);
%     model.cvglmnet_fit{i}.glmnet_fit.npasses = single(model.cvglmnet_fit{i}.glmnet_fit.npasses);
%     model.cvglmnet_fit{i}.glmnet_fit.jerr = single(model.cvglmnet_fit{i}.glmnet_fit.jerr);
%     model.cvglmnet_fit{i}.glmnet_fit.dim = single(model.cvglmnet_fit{i}.glmnet_fit.dim);
%     model.cvglmnet_fit{i}.glmnet_fit.offset = single(model.cvglmnet_fit{i}.glmnet_fit.offset);
% end

%% Load trees on residual
tmp = load('tree_fit_35_1se.mat');
model.tree_fit = tmp.tree_fit;
