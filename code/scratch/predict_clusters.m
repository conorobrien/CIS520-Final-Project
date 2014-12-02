function [X_pred_clust, Y_bins] = predict_clusters(X, Y, X_test, n_bins)

% clust = kmeans([X;X_test], n_clust);
% X_clust = clust(1:size(X,1));
% X_pred_clust = clust((size(X,1)+1):end);

bin_length = floor(length(Y)/n_bins);
Y_bins = zeros(size(Y));
[~, i_sorted] = sort(Y);
for i = 0:(n_bins-2)
    index = i_sorted((i*bin_length +1):((i+1)*bin_length));
    Y_bins(index) = i+1;
end
index = i_sorted((end-bin_length):end);
Y_bins(index) = n_bins;

options = statset('UseParallel',1);

knn = fitcecoc(X, Y_bins, 'Options', options);

X_pred_clust = knn.predict(X_test);