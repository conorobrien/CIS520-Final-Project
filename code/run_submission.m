clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X_train =[word_train bigram_train];
Y_train = price_train;
X_test = [word_test bigram_test];

initialize_additional_features;

%% Run algorithm
% Example by lazy TAs
addpath('liblinear/matlab')

X_pca = spca([X_train; X_test], 150);

X_pca_train = [city_train X_pca(1:length(X_train),:)];
X_pca_test = [city_test X_pca((length(X_train)+1):end,:)];

[Yhat_bins, Y_bins] = predict_bins(X_pca_train, Y_train, X_pca_test, 10);


%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');