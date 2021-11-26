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

# Brain decoding with MLP

## Multilayer Perceptron
```{figure} mlp_decoding/multilayer-perceptron.png
---
width: 800px
name: multilayer-perceptron-fig
---
A multilayer perceptron with 25 units on the input layer, a single hidden layer with 17 units, and an output layer with 9 units. Figure generated with the [NN-SVG](http://alexlenail.me/NN-SVG/index.html) tool by [Alexander Lenail]. The figure is shared under a [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
```
We are going to train a Multilayer Perceptron (MLP) classifier for brain decoding on the Haxby dataset. MLPs are one of the most basic architecture of artificial neural networks. MLPs consist of input and output layers as well as hidden layers that transform the input to the usable data for the output layer. Like other machine learning models for supervised learning, a MLP initially goes through a training phase. During this supervised phase, the network is taught what to look for and what is the desired output.
In this tutorial, we are going to train the simplest MLP architecture featuring one input layer, one output layer and just one hidden layer.

## Getting the data
We are going to download the dataset from Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. You can check section {ref}`haxby-dataset` for more details on that dataset. Here we are going to quickly download it, and prepare it for machine learning applications with a set of predictive variable, the brain time series `X`, and a dependent variable, the annotation on cognition `y`.
```{code-cell} python3
:tags: ["hide_input"]
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
```{code-cell} python3
categories = y.unique()
print(categories)
print(y.shape)
print(X.shape)
```
So we have 1452 time points, with one cognitive annotations each, and for each time point we have recordings of fMRI activity across 675 voxels. We can also see that the cognitive annotations span 9 different categories.

We are going to use Keras for training the MLP, and we are going to convert the string categories into a one-hot encoder:

```{code-cell} python3
# creating instance of one-hot-encoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
enc = OneHotEncoder(handle_unknown='ignore')
y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
# turn the sparse matrix into a pandas dataframe
y = pd.DataFrame(y_onehot.toarray())
display(y)
```

## Training a model
We are going to start by splitting our dataset between train and test. We will keep 20% of the time points as test, and then set up a 10 fold cross validation for training/validation.
```{code-cell} python3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   
```

Now we can build a MLP using Tensorflow and Keras:
```{code-cell} python3
from keras.models import Sequential
from keras.layers import Dense

# number of unique conditions that we have
model_mlp = Sequential()

# Adding the input layer and the first hidden layer
model_mlp.add(Dense(50 , input_dim = 675, kernel_initializer="uniform", activation = 'relu'))

# Adding the second hidden layer
model_mlp.add(Dense(30, kernel_initializer="uniform", activation = 'relu'))

# Using softmax at the end, length of categories shows the number of labels we have
model_mlp.add(Dense(len(categories), activation = 'softmax'))

model_mlp.summary()
```

Time to train that model!
```{code-cell} python3
:tags: ["hide-output"]
# Compiling the model
model_mlp.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the model on the Training set
history = model_mlp.fit(X_train, y_train, batch_size = 10,
                             epochs = 10, validation_split = 0.2)
```
```{code-cell} python3
import sys
sys.path.append('../src')
import visualization
plot_history = visualization.classifier_history (history, 'MLP ')
```

## Assessing performance
Let's check the accuracy of the prediction on the training set:
```{code-cell} python3
# Making the predictions and evaluating the model
from sklearn.metrics import classification_report
y_train_pred = model_mlp.predict(X_train)
print(classification_report(y_train.values.argmax(axis = 1), y_train_pred.argmax(axis=1)))
```
This is dangerously high. Let's check on the test set:
```{code-cell} python3
y_test_pred = model_mlp.predict(X_test)
print(classification_report(y_test.values.argmax(axis = 1), y_test_pred.argmax(axis=1)))
```

We can have a look at the confusion matrix:
```{code-cell} python3
# confusion matrix
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append('../src')
import visualization
cm_svm = confusion_matrix(y_test.values.argmax(axis = 1), y_test_pred.argmax(axis=1))
model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis = 1)[:, np.newaxis]

visualization.conf_matrix(model_conf_matrix,
                          categories,
                          title='MLP decoding results on Haxby')
```

```{warning}
Unfortunately we don't have a simple way to visualize the important features like we did with the linear SVM! You can check this fantastic [distill article](https://distill.pub/2017/feature-visualization/) to learn more about feature visualization in artificial neural networks.
```

## Exercises
 * What is the most difficult category to decode? Why?
 * The model seemed to overfit. Try adding a `Dropout` layer to regularize the model. You can read about dropout in keras in this [blog post](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab).
 * Try to add layers or hidden units, and observe the impact on overfitting and training time.
