load data/city_train.mat
load data/city_test.mat
load data/word_train.mat
load data/word_test.mat
load data/bigram_train.mat
load data/bigram_test.mat
load data/price_train.mat


x_train = [city_train word_train bigram_train];
y_train = price_train;
x_test = [city_test word_test bigram_test];

words = x_train(:,8:5007)';
[U,S,V] = svds(words,3);

scatter3(U(:,1),U(:,2), U(:,3));