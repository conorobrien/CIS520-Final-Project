    clear;

addpath('scratch');

% load ../data/city_train.mat
load ../data/city_test.mat
% load ../data/word_train.mat
load ../data/word_test.mat
% load ../data/bigram_train.mat
load ../data/bigram_test.mat
% load ../data/price_train.mat

% X_train =[city_train word_train bigram_train];
% Y_train = price_train;
X_test = [city_test word_test bigram_test];

% initialize_additional_features;

%% Run algorithm

tic
model = init_model();
fprintf('Initialized Model in %fs\n', toc);

n_preds = size(X_test,1);
prices = zeros(n_preds,1);
tic
t = CTimeleft(n_preds);
for i = 1:n_preds
    prices(i) = make_final_prediction(model, X_test(i,:));
    t.timeleft();
end
fprintf('Made Predictions in %fs\n', toc);

%% Save results to a text file for submission
% dlmwrite('submit_test.txt',prices,'precision','%d');