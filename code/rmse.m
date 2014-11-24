function err = rmse(Y, Yhat)


err = sqrt(mean((Y-Yhat).^2));