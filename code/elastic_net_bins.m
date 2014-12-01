function [ y_hat ] = elastic_net_bins(x_train, y_train, x_test, bins_train, bins_test)
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

for i = 1:K
    x = x_test(bins_test == i, :);
    if ~isempty(x)
        model = models{i};
        y_est = cvglmnetPredict(model, x);
        y_hat(bins_test == i) = y_est;
        disp(['Finished testing on testing bin ', num2str(i)]);
    end
end
end

