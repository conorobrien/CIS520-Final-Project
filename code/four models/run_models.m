load city_train.mat
load city_test.mat
load word_train.mat
load word_test.mat
load bigram_train.mat
load bigram_test.mat
load price_train.mat


x_train = [city_train word_train bigram_train];
y_train = price_train;
x_test = [city_test word_test bigram_test];


%% SVR
% 7 models for 7 cities
% Each uses svr (liblinear package)
yfit_svr = model_svr(x_train, y_train, x_test);

%% PCA
% 7 models for 7 cities
% Each uses elastic net (glmnet package) on semi-supervised PCA'ed data
yfit_pca = model_pca(x_train, y_train, x_test);

%% Kmeans
% 3 clusters
% And trains elastic net (glmnet package) on each cluster
yfit_kmeans = model_kmeans(x_train,y_train,x_test);

%% Elastic Net
% 7 models for 7 cities
% Each uses elastic net (glmnet package) 
yfit_elasticNet = model_elasticnet(x_train,y_train, x_test);

%% Ridge regression and DT Ensemble
% 7 models for 7 cities
% Train ridge regression (glmnet package, set alpha = 0) first, and then use Tree bagger to
% regress on the residual.
% This is the model that we use for the leaderboard.
yfit_ensemble = model_elasticnet_dt_ensemble(x_train,y_train,x_test);