clearvars;

load cvglmnet_fit_old.mat

for i = 1:7
    cvglmnet_fit{i}.glmnet_fit.beta = single(cvglmnet_fit{i}.glmnet_fit.beta);
end

save cvglmnet_fit_small.mat cvglmnet_fit