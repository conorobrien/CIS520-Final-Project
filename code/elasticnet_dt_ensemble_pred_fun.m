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
    % saves space when saving model to disk
    cvglmnet_fit{i}.glmnet_fit.dim = [];
    cvglmnet_fit{i}.glmnet_fit.df = [];
    cvglmnet_fit{i}.lambda = [];
    cvglmnet_fit{i}.cvm = [];
    cvglmnet_fit{i}.cvsd = [];
    cvglmnet_fit{i}.cvup = [];
    cvglmnet_fit{i}.cvlo = [];
    cvglmnet_fit{i}.nzero = [];
    cvglmnet_fit{i}.cvup = [];
    
    y_hat = cvglmnetPredict(cvglmnet_fit{i}, X{i});
    
    % train on the residuals
    residuals{i} = y_hat - Y{i};
    tree_fit{i} = TreeBagger(35, full(X{i}), residuals{i}, 'method', 'regression', 'Options', tree_stats);
    tree_fit{i} = compact(tree_fit{i});
    disp(['trained city # ', num2str(i)]);
end

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

disp('saving models');

for i = 1:7
    cvglmnet_fit{i}.glmnet_fit.beta = single(cvglmnet_fit{i}.glmnet_fit.beta);
end
save('tree_fit_35_1se.mat', 'tree_fit');
save('cvglmnet_fit_1se.mat', 'cvglmnet_fit');
disp('done saving models');

end