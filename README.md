CIS520-Final-Project
====================
Submitted again, with subtracting mean before doing cvgnlmnet fitting, and added the mean after. Interestingly this did slightly worse than the original.

Running svd on the uncentered data, the 3rd pc shows a separated group, which is ads from the same companies, so basically same ads but with different number for bedrooms, etc.

Notes about this cluster:
  -  356 data items
  -  Of these, 166 are 'Lennar' adds
  -  Of these, 15 are 'american west' adds
  -  Of these, 62 are 'richmond american homes' adds
  -  Of these, 185 contain the phrase 'to-be-built'
  -  After removing the 185 'to-be-built' homes, the other 171 of them contain the phrase 'new construction'. That is, every one of the data elements in this cluster are 'to-be-built' or 'new construction' properties!
  
