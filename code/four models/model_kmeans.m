function yfit = model_kmeans(x_train, y_train, x_test)

    stats = statset('UseParallel', true);
    [ind,c] = kmeans([x_train; x_test], 3, 'Distance', 'cityblock','Options', stats);
    
%     save('center.mat','c')
    ind_train = ind(1:numel(y_train));
    ind_test = ind(numel(y_train)+1:end);    
    addpath('glmnet_matlab')

    options = glmnetSet();
    options.alpha = 0;
    
    X{3}=[];
    Y{3}=[];
    fit{3}=[];
    for i = 1:3
        city_idxs = ind_train == i;
        X{i} = x_train(city_idxs, :);
        Y{i} = y_train(city_idxs);
        fit{i} = cvglmnet(X{i}, Y{i}, 'gaussian', options);
        disp(['trained cluster # ', num2str(i)]);
    end
    %save('clust_cvnet.mat','fit')

    yfit = zeros(size(x_test, 1), 1);
    %%
    for i = 1:3
        city_idxs = ind_test == i;
        X{i} = x_test(city_idxs, :);
        yfit(city_idxs)  = cvglmnetPredict( fit{i}, X{i});
    end
    
end