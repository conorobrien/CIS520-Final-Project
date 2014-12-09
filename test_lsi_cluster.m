addpath('code')
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

words = x_train(:,8:end)';
[U,S,V] = svds(words,3);

get_treebagger_splits
clf;
scatter3(U(:, 1),U(:, 2), U(:, 3), 'bo');
for c = 1:numel(top_features)
    idxs = top_features{c};
    switch c
        case 1
            marker = 'r*';
        case 2
            marker = 'g*';
        case 3
            marker = 'b*';
        case 4
            marker = 'c*';
        case 5
            marker = 'y*';
        case 6
            marker = 'm*';
        case 7
            marker = 'k*';
        otherwise
            warning('Unexpected City!!')
    end
    for w = 1:numel(top_features{c})
        feat = idxs(w);
        hold on
        scatter3(U(feat, 1),U(feat, 2), U(feat, 3), marker);
    end
end
% scatter3(U(:,1),U(:,2), U(:,3));