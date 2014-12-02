function prediction = make_final_prediction(model,X_test,~)

addpath('glmnet_matlab');

% Input
% X_test : a 1xp vector representing "1" test sample.
% X_test=[city word bigram] a 1-by-10007 vector
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

city = find(X_test(1:7));

base_fit = cvglmnetPredict(model.cvglmnet_fit{city}, X_test(8:end));
residual_fit = predict(model.tree_fit{city}, full(X_test(8:end)));

prediction = base_fit - residual_fit;
