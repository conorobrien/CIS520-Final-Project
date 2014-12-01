function [yfit] = dt_pred_fun(x_train, y_train, x_test)
tbagger = TreeBagger(500, x_train, y_train, 'Method', 'regression');
yfit = predict(tbagger, x_test);
end
