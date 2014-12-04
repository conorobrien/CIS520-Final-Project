%% control variables
full_prediction =0;
cross_val = 1;
time_prediction = 0;

%% CVGLMNET for different cities, then use a DT forest to fit the residual
load ../../data/city_train.mat
load ../../data/city_test.mat
load ../../data/word_train.mat
load ../../data/word_test.mat
load ../../data/bigram_train.mat
load ../../data/bigram_test.mat
load ../../data/price_train.mat

x_train = [city_train word_train bigram_train];
y_train = price_train;
x_test = [city_test word_test bigram_test];

if cross_val == 1
    rmse = sqrt(crossval('mse', x_train, y_train,'Predfun', ...
        @elasticnet_regression_tree_ensemble_pred_fun, 'kfold', 2));
end

if full_prediction == 1
    yfit = elasticnet_regression_tree_ensemble_pred_fun(x_train, y_train, x_test);
    dlmwrite('submit.txt', yfit)
end

if time_prediction == 1
    disp('loading models')
    tic
    load slow_reg_tree.mat;
    load slow_cvglmnet;
    for i = 1:7
        cvglmnet_fit{i}.glmnet_fit.beta = double(cvglmnet_fit{i}.glmnet_fit.beta);
    end
    toc
    
    disp('testing model')
    tic
    yfit = zeros(size(x_test, 1), 1);
    for i = 1:size(x_test, 1)
        city_idx = find(x_test(i, 1:7) == 1);
        base_fit = cvglmnetPredict(cvglmnet_fit{city_idx}, x_test(i, 8:end));
        residual_fit = predict(tree_fit{city_idx}, full(x_test(i, 8:end)));
        yfit(i) = base_fit - residual_fit;
    end
    toc
end