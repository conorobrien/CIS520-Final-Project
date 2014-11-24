function [Y_pred_bins, Y_bins] = predict_bins(X, Y, X_test, n_bins);

    bin_price = min(Y_train):((max(Y_train) - min(Y_train))/n_bins):max(Y_train);
    Y_bins = zeros(size(Y));
    
    for i = 2:(n_bins+1)
        Y_bins(Y_train>=bin_price(i-1) & Y_train<=bin_price(i)) = i-1;
    end

    svm_model = train(Y_bins,X,[,'-t 3']); %#ok<NBRAK,NOCOM>

    Y_pred_bins = predict(rand(length(X_test),1),X_pca_test, svm_model);

end