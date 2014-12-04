function yfit = model_elasticnet(x_train, y_train, x_test)

    addpath('glmnet_matlab')

    X{7}=[];
    Y{7}=[];
    fit{7}=[];
    for i = 1:7
        city_idxs = x_train(:, i) == 1;
        X{i} = x_train(city_idxs, 8:end);
        Y{i} = y_train(city_idxs);
        fit{i} = cvglmnet(X{i}, Y{i});
        disp(['trained city # ', num2str(i)]);
    end
    yfit = zeros(size(x_test, 1), 1);
    %%
    for i = 1:7
        city_idxs = x_test(:, i) == 1;
        X{i} = x_test(city_idxs, 8:end);
        yfit(city_idxs)  = cvglmnetPredict( fit{i}, X{i});
    end
    
end