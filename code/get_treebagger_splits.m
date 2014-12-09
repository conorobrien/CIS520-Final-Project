% Computes the features most tested from the 35-tree treebagger ensemble.
% top_features is a cell array, one element for each city. top_features{i}
% has the top nwords_per_city words used in the ensemble for city i
load tree_fit_35_1se.mat

nwords_per_city = 100;
ncities = numel(tree_fit);
top_features{ncities} = [];

for c = 1:ncities
    feature_cnt = zeros(1, 10000);
    ntrees = numel(tree_fit{c}.Trees);
    for t = 1:ntrees
        tree = tree_fit{c}.Trees{t};
        cut_pred = tree.CutPredictor;
        
        num_cuts = numel(cut_pred);
        for nc = 1:num_cuts
            feat = cut_pred{nc};
            if numel(feat) == 0
                continue
            end
            feat = feat(2:end);
            feat = str2num(feat);
            feature_cnt(feat) = feature_cnt(feat) + 1;
        end
    end
    
    % a terrible way to find the most used features for each city
    top_feat_idxs = zeros(1, nwords_per_city);
    for i = 1:nwords_per_city
        [~, idx] = max(feature_cnt);
        top_feat_idxs(i) = idx;
        feature_cnt(idx) = -1;
    end
    top_features{c} = top_feat_idxs;
end