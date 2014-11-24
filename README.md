CIS520-Final-Project
====================
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
  
