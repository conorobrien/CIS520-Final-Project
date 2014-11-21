%% Subtract mean of city before doing fit, then add it back after predicting.

% meancity = zeros(7,1);
% ind = zeros(numel(Y_train),1);
% testind = zeros(size(X_test,1),1);
% for i = 1:7
%     meancity(i) = mean(Y_train(X_train(:,i)==1));
%     ind(X_train(:,i)==1) = i;
%     testind(X_test(:,i)==1) = i;
% end
% 
% Ynew = bsxfun(@minus, Y_train, meancity(ind));
% fit = cvglmnet(X_train(:, 8:end), Ynew);
% prices = cvglmnetPredict(fit, X_test(:, 8:end));
% 
% prices = bsxfun(@plus, prices, meancity(testind));

%%
X{7}=[];
Y{7}=[];
fit{7}=[];
for i = 1:7
    X{i} = X_train(  X_train(:,i)==1, 8:end  );
    Y{i} = Y_train( X_train(:,i)==1 );
    fit{i} = cvglmnet(X{i}, Y{i});
end
prices = zeros(size(X_test, 1), 1);
%%
for i = 1:7
    prices(X_test(:,i)==1) = cvglmnetPredict(fit{i}, X_test(X_test(:,i)==1, 8:end));
end