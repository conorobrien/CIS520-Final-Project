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

n_bins = 5;
bin_price = min(Y_train):((max(Y_train) - min(Y_train))/n_bins):max(Y_train);
Y_bin = zeros(size(Y));

for i = 2:(n_bins+1)
    Y_bin(Y_train>=bin_price(i-1) & Y_train<=bin_price(i)) = i-1;
end

svm_model = train(Y_bin,X_pca_train,[,'-t 3']); %#ok<NBRAK,NOCOM>

predicted_bin = predict(rand(length(X_test),1),X_pca_test);



%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');