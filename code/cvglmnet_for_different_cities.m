

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