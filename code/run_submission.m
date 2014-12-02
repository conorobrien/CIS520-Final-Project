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


%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');