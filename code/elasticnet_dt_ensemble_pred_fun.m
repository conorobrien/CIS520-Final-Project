function [yfit] = elasticnet_dt_ensemble_pred_fun(x_train, y_train, x_test)
addpath('glmnet_matlab')


% [~,~,pca_coeff] = svds([x_train(:,8:end); x_test(:,8:end)],200);

X{7} = [];
X_pca{7} = [];
Y{7} = [];
residuals{7} = [];
cvglmnet_fit{7} = [];
tree_fit{7} = [];
res_fit{7} = [];
options = glmnetSet();
options.alpha = 0;
tree_stats = statset('UseParallel', true);
% Split data out by cities. Then, train each city individually
for i = 1:7
    city_idxs = x_train(:, i) == 1;
    X{i} = x_train(city_idxs, 8:end);
%     X_pca{i} = X{i}*pca_coeff;
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
%     tree_fit{i} = TreeBagger(20, full(X{i}), residuals{i}, 'method', 'regression', 'Options', tree_stats, 'minleaf', 10);
%     tree_fit{i} = fitrtree(full(X{i}), residuals{i}, 'QuadraticErrorTolerance',.01);
%     tree_fit{i} = compact(tree_fit{i});
%     res_fit{i} = cvglmnet(X{i},residuals{i}, 'gaussian', options);
%     res_fit{i} = svmtrain(residuals{i}, X{i}, '-s 3 -t 2 -q');
    disp(['trained city # ', num2str(i)]);
end

Xtree = [];
resTree = [];
for i = 1:7
    Xtree = [Xtree; X{i}];
    resTree = [resTree; residuals{i}];
end

tree_fit = TreeBagger(20, full(Xtree), resTree, 'method', 'regression', 'Options', tree_stats);
% generate prediction based on the elastic net and residual computations
yfit = zeros(size(x_test, 1), 1);
for i = 1:7
    city_idxs = x_test(:, i) == 1;
    X{i} = x_test(city_idxs, 8:end);
%     X_pca{i} = X{i}*pca_coeff;
    base_fit = cvglmnetPredict(cvglmnet_fit{i}, X{i});
    residual_fit1 = predict(tree_fit, full(X{i}));
%     residual_fit1 = svmpredict(X{i}(:,1), X{i},res_fit{i});
%     residual_fit2 = cvglmnetPredict(res_fit{i}, X{i}, 'lambda_min');
%     yfit(city_idxs) = base_fit - (residual_fit1+residual_fit2)/2;
    yfit(city_idxs) = base_fit - residual_fit1;
end
disp('done testing')

disp('saving models');

% for i = 1:7
%     cvglmnet_fit{i}.glmnet_fit.beta = single(cvglmnet_fit{i}.glmnet_fit.beta);
% end
% save('tree_fit_35_1se.mat', 'tree_fit');
% save('cvglmnet_fit_1se.mat', 'cvglmnet_fit');
% disp('done saving models');

end