function yfit = model_svr_pca(x_train, y_train, x_test)
    addpath('liblinear/matlab')
    
    [~,~,V] = svds([x_train(:, 8:end); x_test(:, 8:end)], 100);
    
    X{7} = [];
    Y{7} = [];
    svr_fit{7} = [];

    for i = 1:7
        city_idxs = x_train(:, i) == 1;
        X{i} = x_train(city_idxs, 8:end)*V;
        Y{i} = y_train(city_idxs);
        svr_fit{i} = train(Y{i}, X{i}, '-s 11');
        disp(['trained city # ', num2str(i)]);
    end

    yfit = zeros(size(x_test, 1), 1);
    for i = 1:7
        city_idxs = x_test(:, i) == 1;
        X{i} = x_test(city_idxs, 8:end)*V;
        yfit(city_idxs) = predict(zeros(size(X{i},1), 1),  X{i}, svr_fit{i}, '-q');
    end

end