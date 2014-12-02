clearvars

addpath('liblinear/matlab')
addpath('libsvm');

load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat
% 
load X_pca
X_pca = X_pca(:,1:200);
X_pca = zscore(X_pca);
X_pca_city = full([city_train X_pca]);
Y = full(price_train);

n_bins = 10;
bin_length = floor(length(Y)/n_bins);
Y_bins = zeros(size(Y));
[~, i_sorted] = sort(Y);
for i = 0:(n_bins-2)
    index = i_sorted((i*bin_length +1):((i+1)*bin_length));
    Y_bins(index) = i+1;
end
index = i_sorted((end-bin_length):end);
Y_bins(index) = n_bins;

K = round(logspace(0,log10(150),5)); % number of neighbors
cvloss = zeros(numel(K),1);
for k=1:numel(K)
    knn = fitcknn(X_pca_city,Y_bins,...
        'NumNeighbors',K(k),'CrossVal','On');
    cvloss(k) = kfoldLoss(knn);
    k
end
figure; % Plot the accuracy versus k
semilogx(K,cvloss);
xlabel('Number of nearest neighbors');
ylabel('10 fold classification error');
title('k-NN classification');

beep