function yfit = model_pca(x_train,y_train, x_test)

    [~,~,V] = svds([x_train(:,8:end); x_test(:, 8:end)], 50);

    addpath('glmnet_matlab')

    X{7}=[];
    Y{7}=[];
    fit{7}=[];
    for i = 1:7
        city_idxs = x_train(:, i) == 1;
        X{i} = x_train(city_idxs, 8:end)*V;
        Y{i} = y_train(city_idxs);
        fit{i} = cvglmnet(X{i}, Y{i});
        disp(['trained city # ', num2str(i)]);
    end
    yfit = zeros(size(x_test, 1), 1);

    for i = 1:7
        city_idxs = x_test(:, i) == 1;
        X{i} = x_test(city_idxs, 8:end)*V;
        yfit(city_idxs)  = cvglmnetPredict( fit{i}, X{i});
    end
    
end