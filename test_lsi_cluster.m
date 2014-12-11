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

not_top_city_word = ones(1, size(words, 1));
for c = 1:numel(top_features)
    not_top_city_word(top_features{c}) = 0;
end
rvec = rand(size(not_top_city_word));
not_top_city_word(rvec > .99) = 0;
not_top_city_word = logical(not_top_city_word);
scatter3(U(not_top_city_word, 1),U(not_top_city_word, 2), U(not_top_city_word, 3), 'b.');

for c = 1:numel(top_features)
    idxs = top_features{c};
    switch c
        case 1
            marker = 'ro';
        case 2
            marker = 'go';
        case 3
            marker = 'bo';
        case 4
            marker = 'co';
        case 5
            marker = 'yo';
        case 6
            marker = 'mo';
        case 7
            marker = 'ko';
        otherwise
            warning('Unexpected City!!')
    end
    for w = 1:numel(top_features{c})
        feat = idxs(w);
        hold on
        scatter3(U(feat, 1),U(feat, 2), U(feat, 3), marker);
    end
end
xlabel('LSI Component 1', 'FontSize', 24)
ylabel('LSI Component 2', 'FontSize', 24)
zlabel('LSI Component 3', 'FontSize', 24)
title('LSI Projection', 'FontSize', 24)