function [Y_pred_bins, Y_bins] = predict_bins(X, Y, X_test, n_bins)
    addpath('liblinear/matlab')

    bin_price = min(Y):((max(Y) - min(Y))/n_bins):max(Y);
    Y_bins = zeros(size(Y));
    
    for i = 2:(n_bins+1)
        Y_bins(Y>=bin_price(i-1) & Y<=bin_price(i)) = i-1;
    end

    svm_model = train(Y_bins,X,[,'-s 1']); %#ok<NBRAK,NOCOM>

    Y_pred_bins = predict(rand(length(X_test),1),X_test, svm_model);

end