function [yfit] = elasticnet_dt_ensemble_pred_fun(x_train, y_train, x_test)
addpath('glmnet_matlab')

X{7} = [];
Y{7} = [];
residuals{7} = [];
cvglmnet_fit{7} = [];
tree_fit{7} = [];
options = glmnetSet();
options.alpha = 0;
tree_stats = statset('UseParallel', true);
% Split data out by cities. Then, train each city individually
for i = 1:7
    city_idxs = x_train(:, i) == 1;
    X{i} = x_train(city_idxs, 8:end);
    Y{i} = y_train(city_idxs);
    cvglmnet_fit{i} = cvglmnet(X{i}, Y{i}, 'gaussian', options);
    y_hat = cvglmnetPredict(cvglmnet_fit{i}, X{i});
    
    % train on the residuals
    residuals{i} = y_hat - Y{i};
    tree_fit{i} = TreeBagger(20, full(X{i}), residuals{i}, 'method', 'regression', 'Options', tree_stats);
    i
end
disp('done training')

% generate prediction based on the elastic net and residual computations
yfit = zeros(size(x_test, 1), 1);
for i = 1:7
    city_idxs = x_test(:, i) == 1;
    X{i} = x_test(city_idxs, 8:end);

    base_fit = cvglmnetPredict(cvglmnet_fit{i}, X{i});
    residual_fit = predict(tree_fit{i}, full(X{i}));
    yfit(city_idxs) = base_fit - residual_fit;
end
disp('done testing')
end