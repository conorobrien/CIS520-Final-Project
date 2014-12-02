addpath('glmnet_matlab')

%% Bins data into K bins, then regresses on each bin using elastic net

load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

K = 5;

x_train =[word_train bigram_train city_train];
x_train = spca(x_train, 50); 
y_train = price_train;

y_hat = zeros(size(y_train));
y_train_class = zeros(size(y_train));

min_y = floor(min(y_train));
max_y = ceil(max(y_train));
bin_size = (max_y - min_y)/K;

bin_idx{K} = [];

for i = 1:K
    lower_bound = (i - 1) * bin_size + min_y;
    upper_bound = lower_bound + bin_size;
    bin_idx{i} = find(y_train >= lower_bound & y_train < upper_bound); 
end

% y_train_class = the class, as determined by price bracket
for i = 1:K
    y_train_class(bin_idx{i}) = i;
end
if false
   nb = NaiveBayes.fit(full(x_train), y_train_class, 'Distribution', 'mvmn');
   y_pred_class = predict(nb, x_train);
else
    model = cvglmnet(x_train, y_train_class, 'gaussian');
    y_pred_class = cvglmnetPredict(model, x_train);
    y_pred_class = round(y_pred_class);
    y_pred_class(y_pred_class > K) = K;
    y_pred_class(y_pred_class < 1) = 1;
end


for i = 1:K
    x = x_train(bin_idx{i}, :);
    y = y_pred_class(bin_idx{i});
    model = cvglmnet(x, y, 'gaussian');
    y_est = cvglmnetPredict(model, x);
    y_hat(bin_idx{i}) = y_est;
    disp(['Finished estimating bin ', num2str(i)]);
end

y_hat = round(y_hat);
y_hat(y_hat > K) = K;
y_hat(y_hat < 1) = 1;
y_hat_prices = zeros(size(y_hat));
for i = 1:K
    idx_into_prices = (y_train_class == i);
    mean_price = mean(y_train(idx_into_prices));
    y_hat_prices(y_hat == i) = mean_price;
end
y_hat = y_hat_prices;

mse = sqrt((sum(((y_hat - y_train) .^ 2))/numel(y_train)))
misclass_rate = sum(y_pred_class ~= y_train_class)/ numel(y_train_class)

figure(1); clf;
scatter(y_train, y_hat - y_train)
figure(2); clf;
scatter(y_train(y_pred_class == 2), y_hat(y_pred_class == 2) - y_train(y_pred_class == 2))
