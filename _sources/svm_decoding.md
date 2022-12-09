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
  display_name: Python 3
  language: python
  name: python3
---

# Brain decoding with SVM

## Support vector machines
```{figure} svm_decoding/optimal-hyperplane.png
---
width: 500px
name: optimal-hyperplane-fig
---
A SVM aims at finding an optimal hyperplane to separate two classes in high-dimensional space, while maximizing the margin. Image from the [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html) under [BSD 3-Clause license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
```
We are going to train a support vector machine (SVM) classifier for brain decoding on the Haxby dataset. SVM is often successful in high dimensional spaces, and it is a popular technique in neuroimaging.

In the SVM algorithm, we plot each data item as a point in N-dimensional space that N depends on the number of features that distinctly classify the data points (e.g. when the number of features is 3 the hyperplane becomes a two-dimensional plane.). The objective here is finding a hyperplane (decision boundaries that help classify the data points) with the maximum margin (i.e the maximum distance between data points of both classes). Data points falling on either side of the hyperplane can be attributed to different classes.

The scikit-learn [documentation](https://scikit-learn.org/stable/modules/svm.html) contains a detailed description of different variants of SVM, as well as example of applications with simple datasets.

## Getting the data
We are going to download the dataset from Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. You can check section {ref}`haxby-dataset` for more details on that dataset. Here we are going to quickly download it, and prepare it for machine learning applications with a set of predictive variable, the brain time series `X`, and a dependent variable, the annotation on cognition `y`.

```{code-cell} ipython3
import os
import warnings
warnings.filterwarnings(action='once')

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

```{code-cell} ipython3
categories = y.unique()
print(categories)
print(y.shape)
print(X.shape)
```

So we have 1452 time points, with one cognitive annotations each, and for each time point we have recordings of fMRI activity across 675 voxels. We can also see that the cognitive annotations span 9 different categories.

## Training a model
We are going to start by splitting our dataset between train and test. We will keep 20% of the time points as test, and then set up a 10 fold cross validation for training/validation.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   
```

Now we can initialize a SVM classifier, and train it:

```{code-cell} ipython3
from sklearn.svm import SVC
model_svm = SVC(random_state=0, kernel='linear', C=1)
model_svm.fit(X_train, y_train)
```

## Assessing performance
Let's check the accuracy of the prediction on the training set:

```{code-cell} ipython3
from sklearn.metrics import classification_report
y_train_pred = model_svm.predict(X_train)
print(classification_report(y_train, y_train_pred))
```

This is dangerously high. Let's check on the test set:

```{code-cell} ipython3
y_test_pred = model_svm.predict(X_test)
print(classification_report(y_test, y_test_pred))
```

We can have a look at the confusion matrix:

```{code-cell} ipython3
# confusion matrix
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append('../src')
import visualization
cm_svm = confusion_matrix(y_test, y_test_pred)
model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]

visualization.conf_matrix(model_conf_matrix,
                          categories,
                          title='SVM decoding results on Haxby')
```

## Visualizing the weights
Finally we can visualize the weights of the (linear) classifier to see which brain region seem to impact most the decision, for example for faces:

```{code-cell} ipython3
from nilearn import plotting
# first row of coef_ is comparing the first pair of class labels
# with 9 classes, there are 9 * 8 / 2 distinct
coef_img = masker.inverse_transform(model_svm.coef_[0, :])
plotting.view_img(
    coef_img, bg_img=haxby_dataset.anat[0],
    title="SVM weights", dim=-1, resampling_interpolation='nearest'
)
```

## And now the easy way
We can use the high-level `Decoder` object from Nilearn. See [Decoder object](https://nilearn.github.io/dev/modules/generated/nilearn.decoding.Decoder.html) for details. It reduces model specification and fit to two lines of code:

```{code-cell} ipython3
from nilearn.decoding import Decoder
# Specify the classifier to the decoder object.
# With the decoder we can input the masker directly.
# We are using the svc_l1 here because it is intra subject.
#
# cv=5 means that we use 5-fold cross-validation
#
# As a scoring scheme, one can use f1, accuracy or ROC-AUC
#
decoder = Decoder(estimator='svc', cv=5, mask=mask_filename, scoring='f1') 
decoder.fit(func_file, y)
```

That's it !
We can now look at the results: F1 score and coefficient image:

```{code-cell} ipython3
print('F1 scores')
for category in categories:
    print(category, '\t\t    {:.2f}'.format(np.mean(decoder.cv_scores_[category])))
plotting.view_img(
    decoder.coef_img_['face'], bg_img=haxby_dataset.anat[0],
    title="SVM weights for face", dim=-1, resampling_interpolation='nearest'
)
```

Note: the Decoder implements a one-vs-all strategy. Note that this is a better choice in general than one-vs-one.

## Getting more meaningful weight maps with Frem
It is often tempting to interpret regions with high weights as 'important' for the prediction task. However, there is no statistical guarantee on these maps. Moreover, they iften do not even exhibit very clear structure. To improve that, a regularization can be brought by using the so-called Fast Regularized Ensembles of models (FREM), that rely on simple averaging and clustering tools to provide smoother maps, yet with minimal computational overhead.

```{code-cell} ipython3
from nilearn.decoding import FREMClassifier
frem = FREMClassifier(estimator='svc', cv=5, mask=mask_filename, scoring='f1')
frem.fit(func_file, y)
plotting.view_img(
    frem.coef_img_['face'], bg_img=haxby_dataset.anat[0],
    title="SVM weights for face", dim=-1, resampling_interpolation='nearest'
)
```

Note that the resulting accuracy is in general slightly higher:

```{code-cell} ipython3
print('F1 scoreswith FREM')
for category in categories:
    print(category, '\t\t    {:.2f}'.format(np.mean(decoder.cv_scores_[category])))
```

## Exercises
 * What is the most difficult category to decode? Why?
 * The model seemed to overfit. Can you find a parameter value for `C` in `SVC` such that the model does not overfit as much?
 * Try a `'rbf'` kernel in `SVC`. Can you get a better test accuracy than with the `'linear'` kernel?
 * Try to explore the weights associated with other labels.
 * Instead of doing a 5-fold cross-validation, on should split the data by runs.
Implement a leave-one-run and leave-two-run out cross-validation. For that you will need to access the run information, that is stored in `behavioral[chunks]`. You will also need the [LeavePGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut) object of scikit-learn.
 * Try implementing a random forest or k nearest neighbor classifier.
 * **Hard**: implement a systematic hyper-parameter optimization using nested cross-validation. Tip: check this [scikit-learn tutorial](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py).
 * **Hard**: try to account for class imbalance in the dataset.
