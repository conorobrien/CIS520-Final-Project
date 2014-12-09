clearvars

city = 4;

load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X = [city_train word_train bigram_train];
Y = price_train;

X_city = X(X(:,city) == 1, :);
Y_city = Y(X(:,city) == 1, :);
xval_part = make_xval_partition(numel(Y_city), 10);

alpha_err = [];
err = [];

for alpha = 0:0.02:.1
    for i = 1:10
        x_train = X_city(xval_part ~= i,:);
        y_train = Y_city(xval_part ~= i);
        x_test = X_city(xval_part == i,:);
        y_test = Y_city(xval_part == i);

        y_hat = alpha_pred_fun(x_train, y_train, x_test, alpha);
        
        err(i) = rmse(y_hat, y_test);
    end
    alpha_err(end+1,1:2) = [mean(err) alpha];
end
city
alpha_err

    