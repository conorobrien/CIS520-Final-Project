function [yfit] = alpha_pred_fun(x_train, y_train, x_test, varargin)
addpath('glmnet_matlab')

options = glmnetSet();
if ~isempty(varargin)
    options.alpha = varargin{1};
else
    options.alpha = 0;
end

cvglmnet_fit = cvglmnet(x_train(:,8:end), y_train, 'gaussian', options);
yfit = cvglmnetPredict(cvglmnet_fit, x_test);

disp('done testing')

end