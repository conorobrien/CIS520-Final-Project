clear;
load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

X_raw =[city_train word_train bigram_train];
Y_raw = price_train;

xval_part = make_xval_partition(length(Y_train), 10)

for i = 1:10

	X_train = X_raw(xval_part ~= i, :);
	Y_train = Y_raw(xval_part ~= i, :);
	X_test = X_raw(xval_part == i, :);
	Y_test = Y_raw(xval_part == i, :);

	initialize_additional_features;

	%% Run algorithm
	% Example by lazy TAs


	err(i) = mean(Y_pred ~= Y_test)

end




%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');