clear all
load ../data/city_test.mat
load ../data/word_test.mat
load ../data/bigram_test.mat

X_test = [city_test word_test bigram_test];

model = init_model();

for i = 1:1000
    make_final_prediction(model, X_test(i,:));
end