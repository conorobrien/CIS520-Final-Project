function [ rmse ] = sequentialfs_fun(x_train, y_train, x_test, y_test)
options = glmnetSet();
options.alpha = 0;
model = cvglmnet(x_train, y_train, 'gaussian', options);
y_hat = cvglmnetPredict(model, x_test);
rmse = sqrt(sum(((y_hat - y_test) .^ 2)) / numel(y_test));
end