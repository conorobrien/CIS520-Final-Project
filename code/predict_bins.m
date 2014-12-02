function [Y_pred_bins, Y_bins] = predict_bins(X, Y, X_test, n_bins)
    addpath('liblinear/matlab')

%     bin_length = floor(length(Y)/n_bins);
%     Y_bins = zeros(size(Y));
%     [~, i_sorted] = sort(Y);
%     for i = 0:(n_bins-2)
%         index = i_sorted((i*bin_length +1):((i+1)*bin_length));
%         Y_bins(index) = i+1;
%     end
%     index = i_sorted((end-bin_length):end);
%     Y_bins(index) = n_bins;

    bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
    Y_bins = zeros(size(Y));

    for i = 2:(n_bins+1)
        Y_bins(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
    end    
%% LIBLINEAR   
%     svm_model = train(Y_bins,X,[,'-s 6 -q']); %#ok<NBRAK,NOCOM>
%     Y_pred_bins1 = predict(rand(size(X_test,1),1),X_test, svm_model, '-q');
%% KNN Ensemble
%     learner = templateKNN('NumNeighbors',25);
%     GBEnsemble = fitensemble(X,Y_bins,'AdaBoostM1',1000, learner,'type', 'classification');%,'NPredToSample',10);
%     Y_pred_bins = GBEnsemble.predict(X_test);
%% Decision tree Ensemble
%     learner = templateTree();
%     GBEnsemble = fitensemble(X,Y_bins,'TotalBoost',200, learner,'type', 'classification');%,'NPredToSample',10);
%     Y_pred_bins = GBEnsemble.predict(X_test);
%% NB
%     nb = NaiveBayes.fit(X(:,8:end), Y_bins);
%     Y_pred_bins = predict(nb, X_test(:,8:end));
%% GLMNET
%     model = cvglmnet(X, Y_bins, 'binomial');
%     Y_pred_bins = round(cvglmnetPredict(model, X_test));
%% LIBSVM
%     svm_model = svmtrain(Y_bins,X,[,'-s 0 -t 2 -q']); %#ok<NBRAK,NOCOM>
%     Y_pred_bins = svmpredict(rand(size(X_test,1),1),X_test, svm_model);
%% Ensemble SVM
%     mdl = fitcecoc(X, Y_bins);
%     Y_pred_bins = predict(mdl, X_test);
% Y_pred_bins = round(mean([ Y_pred_bins2 Y_pred_bins3], 2));
end