
addpath glmnet_matlab
load ../data/city_test.mat
load ../data/word_test.mat
load ../data/bigram_test.mat

x_test = [city_test word_test bigram_test];

model = init_model();

prices = zeros(1,200);

for i = 1:200
    prices(i) = make_final_prediction(model, x_test(i,:), []);
end
