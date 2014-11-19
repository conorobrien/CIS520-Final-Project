addpath('../original_text/')
addpath('glmnet_matlab')
%%
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat
%% control variables
m11 = 0;
m12 = 1;
%% Running stepwiselm on PCA'd data
if m11 == 1
    x_train =[word_train bigram_train];
    y_train = price_train;

    [u, s, v] = svds(x_train, 6);
    x_new_coord = x_train * v;
    % scatter3(x_new_coord(:, 2), x_new_coord(:, 3), y_train)
    % xlabel('comp2'); ylabel('comp3'); zlabel('price');

    pca_x = [x_new_coord city_train];
    mdl = stepwiselm(pca_x, y_train);
    y_hat = predict(mdl, pca_x);

    % rms calculation
    rms = sqrt(sum(((y_hat - y_train) .^ 2)/numel(y_train)));
end
%% cvglmnet
if m12 == 1
    x_train =[word_train bigram_train city_train];
    y_train = price_train;

    alpha_vals = 0:.1:1;
    best_alpha = 0;
    best_model = 0;
    best_rmse = 100000;
    
    for i = 1:numel(alpha_vals);
        disp(['Testing alpha = ', num2str(alpha_vals(i))]);
        options = glmnetSet;
        options.alpha = alpha_vals(i);	%mixture param

        model = cvglmnet(x_train, y_train, 'gaussian', options);
        y_hat = cvglmnetPredict(model, x_train);
        rmse = sqrt(sum(((y_hat - y_train) .^ 2)/numel(y_train)));
        if rmse < best_rmse
            best_alpha = options.alpha;
            best_model = model;
            best_rmse = rmse;
        end
    end
    disp(['Best alpha value: ', num2str(best_alpha)]);
    disp(['Best rmse: ', num2str(best_rmse)]);
end