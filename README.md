CIS520-Final-Project
====================
Using 20 trees with [city word bigram], it takes 390 seconds to finish all test data on biglab. The leaderboard error is above 0.68. So we don't have much room to add trees.

Using 25 trees, it takes 450 seconds. Using 1 tree, it takes 215 seconds. It seems that the tree running time is linear with number of trees, so we can use about 40 trees at most.

Running 2-fold cv on numbers of trees on biglab, the mse does have several local minima. I ran 15-75 with increment of 10, and by now 75 seems to be the best. 85 is still running.

Using just words alone with 120 trees gave slightly worse prediction accuracy, however, it is still below .68.

The elastic net with decesion tree on the residual with 75 trees gives us a test error of 0.6770, which as of Dec 1, is the top of the board. I'm playing around with saving space with the cvglmnet model file.

Ran KNN (xvalidated) on top 400 principle components + cities. MSE = 1.75. Ran TreeBagger (xvalidated) to create an Ensemble of decesion trees (500 trees on the top 400 principle components). MSE = .7836, which was better. I tried running it with cities as well, but it crashes matlab. This may be a path worth pursuing. I've saved these models locally only (due to size constraint).


Ran crossvalidation on PCAed data with svr and cvglmnet. Not surprisingly, using 2000 pc's would give the best error rate, but still worse than using original data, which is much faster.

Added the first 1000 loadings. Premultiply by X (without the cities) to get the scores. I actually have 2000 loadings but the file is too big for github.

Run this command in biglab if you want it in background.
nohup matlab -nodesktop -nodisplay < file.m > result.txt &

Used alpha=0, ie, ridge regression, error 0.7316. Did cv, alpha=0 was best in cv training error, alpha=0.2 was second best. Don't know if alpha=0 is overfitting, though.

Used 50 PC's from pca, error above 0.8. Then did a little cross-validation for pca till 150 pc's, with the cv training error still decreasing. It took a long time on my laptop. We definitely need to use biglab if we want to see how pca does.

Used 7 fits for 7 cities, error 0.7376. Considering PCA before fitting. Lyle mentioned 40 pcs might be able to capture 90% of variance.

Submitted again, with subtracting mean before doing cvgnlmnet fitting, and added the mean after. Interestingly this did slightly worse than the original.

Running svd on the uncentered data, the 3rd pc shows a separated group, which is ads from the same companies, so basically same ads but with different number for bedrooms, etc.

Notes about this cluster:
  -  356 data items
  -  Of these, 166 are 'Lennar' adds
  -  Of these, 15 are 'american west' adds
  -  Of these, 62 are 'richmond american homes' adds
  -  Of these, 185 contain the phrase 'to-be-built'
  -  After removing the 185 'to-be-built' homes, the other 171 of them contain the phrase 'new construction'. That is, every one of the data elements in this cluster are 'to-be-built' or 'new construction' properties!
  
