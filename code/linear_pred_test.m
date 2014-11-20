function [ Yhat ] = linear_pred_test(X, Y, Xtest )

w = (X'*X)\X'*Y;
Yhat = Xtest*w;

end
