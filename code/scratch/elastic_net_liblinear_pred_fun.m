function [yfit] = elastic_net_liblinear_pred_fun(x_train, y_train, x_test)
addpath('../glmnet_matlab')
addpath('../liblinear/matlab')

X{7} = [];
Y{7} = [];
yfit = zeros(size(x_test, 1), 1);
residuals = zeros(size(yfit));

base_fit{7} = [];
resid_fit{7} = [];
options = glmnetSet();
options.alpha = 0;
% Split data out by cities. Then, train each city individually
for i = 1:7
    city_idxs = x_train(:, i) == 1;
    X{i} = x_train(city_idxs, 8:end);
    Y{i} = y_train(city_idxs);
    base_fit{i} = cvglmnet(X{i}, Y{i}, 'gaussian', options);
    y_hat = cvglmnetPredict(base_fit{i}, X{i});
    
    % train on the residuals
    residuals(city_idxs) = y_hat - Y{i};
    resid_fit{i} = train(residuals(city_idxs), X{i}, [,'-s 11']); %#ok<NBRAK,NOCOM>
    
    disp(['trained city # ', num2str(i)]);
end

for i = 1:7
    city_idxs = x_test(:, i) == 1;
    X{i} = x_test(city_idxs, 8:end);
    base = cvglmnetPredict(base_fit{i}, X{i});
    dummy = zeros(size(X{i}, 1), 1);
    dummy = double(dummy);
    resid = predict(dummy, X{i}, resid_fit{i});
    yfit(city_idxs) = base - resid;
end
disp('done testing')

% disp('saving models');
% save('rbf_elastic_net_base.mat', 'base_fit');
% save('rbf_elastic_net_residual.mat', 'resid_fit');
% disp('done saving models');

end