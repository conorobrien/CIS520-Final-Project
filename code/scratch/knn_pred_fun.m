function [yfit] = knn_pred_fun(x_train, y_train, x_test)
knn = fitcknn(x_train, y_train);
yfit = predict(knn, x_test);
end
