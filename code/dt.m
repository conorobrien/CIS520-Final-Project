load ../data/city_train.mat
load ../data/city_test.mat
load ../data/price_train.mat
load X_pca

x_train = [X_pca];
y_train = price_train;
mse = crossval('mse', x_train, y_train,'Predfun', @dt_pred_fun, 'kfold', 10)
