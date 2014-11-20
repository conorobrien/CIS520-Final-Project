function [score] = spca(X,n)

[U,S,~] = svds(X, n);

score = U*S;