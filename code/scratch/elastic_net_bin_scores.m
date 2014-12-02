function [ y_hat ] = elastic_net_bins(x_train, y_train, x_test, bins_train, bin_scores)
%Given X_TRAIN, Y_TRAIN, and a vector of BINS (assignment to bins), run an
%elastic net model and return the prediction
addpath('glmnet_matlab')

K = max(bins_train);
models{K} = [];
y_hat = zeros(size(x_test, 1), 1);

for i = 1:K
    x = x_train(bins_train == i, :);
    y = y_train(bins_train == i);
    models{i} = cvglmnet(x, y, 'gaussian');
    disp(['Finished training on training bin ', num2str(i)]);
end

% y_est = 
for i = 1:K
    model = models{i};
    y_est(:,i) = cvglmnetPredict(model, x_test);
    disp(['Finished testing on testing bin ', num2str(i)]);
end

y_hat = sum(y_est.*bin_scores, 2);
end

