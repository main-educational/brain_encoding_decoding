---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Brain decoding with SVM

## Support vector machines
We are going to train a support vector machine (SVM) classifier for brain decoding on the Haxby dataset. SVM is often successful in high dimensional spaces, and it is a popular technique in neuroimaging.

In the SVM algorithm, we plot each data item as a point in N-dimensional space that N depends on the number of features that distinctly classify the data points (e.g. when the number of features is 3 the hyperplane becomes a two-dimensional plane.). The objective here is finding a hyperplane (decision boundaries that help classify the data points) with the maximum margin (i.e the maximum distance between data points of both classes). Data points falling on either side of the hyperplane can be attributed to different classes.

The scikit-learn [documentation](https://scikit-learn.org/stable/modules/svm.html) contains a detailed description of different variants of SVM, as well as example of applications with simple datasets.

## Getting the data
We are going to download the dataset from Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. You can check section {ref}`haxby-dataset` for more details on that dataset. Here we are going to quickly download it, and prepare it for machine learning applications with a set of predictive variable, the brain time series `X`, and a dependent variable, the annotation on cognition `y`.
```{code-cell} python3
:tags: ["hide_input"]
import os
from nilearn import datasets
# We are fetching the data for subject 4
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

# mask the data
from nilearn.input_data import NiftiMasker
mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
X = masker.fit_transform(func_file)

# cognitive annotations
import pandas as pd
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
y = behavioral['labels']
```

Let's check the size of `X` and `y`:
```{code-cell} python3
categories = y.unique()
print(categories)
print(y.shape)
print(X.shape)
```
So we have 1452 time points, with one cognitive annotations each, and for each time point we have recordings of fMRI activity across 675 voxels. We can also see that the cognitive annotations span 9 different categories.

Before we carry on with the brain decoding, we first need to convert the cognitive annotations to some numeric values:
```{code-cell} python3
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Encoding the string to numerical values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y = y.ravel()
print(np.unique(y))
```

## Training a model
We are going to start by splitting our dataset between train and test. We will keep 20% of the time points as test, and then set up a 10 fold cross validation for training/validation.
```{code-cell} python3
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   

# prepare the cross-validation procedure
cv = KFold(n_splits = 10, random_state = 0, shuffle = True)
```

Now we can initialize a SVM classifier, and train it:
```{code-cell} python3
from sklearn.svm import SVC
model_svm = SVC(decision_function_shape = 'ovo', random_state = 0, kernel='linear')
model_svm.fit(X_train, y_train)
model_svm.report
```
