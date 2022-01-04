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

# Brain decoding with GCN

## Graph Convolution Neural network
```{figure} gcn_decoding/GCN_pipeline.png
---
width: 500px
name: gcn-pipeline-fig
---
```

- __6__ graph convolutional layers
- __32 graph filters__  at each layer
- followed by a __global average pooling__ layer
- __2 fully connected__ layers 

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

```{code-cell} ipython3
conn_path = os.path.join(data_dir, 'haxby_connectomes/')
if not os.path.exists(conn_path):
    os.makedirs(conn_path)

import glob
import nilearn.connectome
import numpy as np
import warnings
warnings.filterwarnings(action='once')
# Estimating connectomes and save for pytorch to load
corr_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
conn = corr_measure.fit_transform([X])[0]
np.save(os.path.join(conn_path, 'conn_subj{}.npy'.format(sub_no)), conn)
```

```{code-cell} ipython3
concat_path = os.path.join(data_dir, 'haxby_concat/')
if not os.path.exists(concat_path):
    os.makedirs(concat_path)

concat_bold_files = []
for i in range(0,len(y)):
    label = y[i]
    concat_bold_files = X[i:i+1]
    concat_file_name = concat_path + '{}_concat_fMRI.npy'.format(label)
    
    if os.path.isfile(concat_file_name):
        concat_file = np.load(concat_file_name, allow_pickle = True)
        concat_file = np.concatenate((concat_file, concat_bold_files), axis = 0)
        np.save(concat_file_name, concat_file)
    else:
        np.save(concat_file_name, concat_bold_files)



import pandas as pd
split_path = os.path.join(data_dir, 'haxby_split_win/')
if not os.path.exists(split_path):
    os.makedirs(split_path)

dic_labels = {'rest':0,'face':1,'chair':2,'scissors':3,'shoe':4,'scrambledpix':5,'house':6,'cat':7,'bottle':8}
label_df = pd.DataFrame(columns=['label', 'filename'])
processed_bold_files = sorted(glob.glob(concat_path + '/*.npy'))
window_length = 1
out_file = os.path.join(split_path, '{}_{:04d}.npy')
out_csv = os.path.join(split_path, 'labels.csv')

for proc_bold in processed_bold_files:
    
    ts_data = np.load(proc_bold)
    ts_duration = len(ts_data)

    ts_filename = os.path.basename(proc_bold)
    ts_label = ts_filename.split('_', 1)[0]

    valid_label = dic_labels[ts_label]
    
    # Split the timeseries
    rem = ts_duration % window_length
    n_splits = int(np.floor(ts_duration / window_length))

    ts_data = ts_data[:(ts_duration-rem), :]   
    
    for j, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file_name = out_file.format(ts_filename, j)

        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)
        curr_label = {'label': valid_label, 'filename': os.path.basename(ts_output_file_name)}
        label_df = label_df.append(curr_label, ignore_index=True)
    
label_df.to_csv(out_csv, index=False)  
```

```{code-cell} ipython3
# split dataset
import sys
sys.path.append('../src')
from gcn_windows_dataset import TimeWindowsDataset

random_seed = 0

train_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="train", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

valid_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="valid", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

test_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="test", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

print("train dataset: {}".format(train_dataset))
print("valid dataset: {}".format(valid_dataset))
print("test dataset: {}".format(test_dataset))
```

```{code-cell} ipython3
import torch

torch.manual_seed(random_seed)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_generator = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
train_features, train_labels = next(iter(train_generator))
print(f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
print(f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")
```

## Building brain graphs

After loading brain connectome, we will build brain garph.
__k-Nearest Neighbours(KNN) graph__ for the group average connectome will be built based on the connectivity-matrix.

Each node is only connected to *k* other neighbours, which is __8 nodes__ with the strongest regions connectivity in this experiment.

For more details you please check out __*src/graph_construction.py*__ script.

```{code-cell} ipython3
conn_files = sorted(glob.glob(conn_path + '/*.npy')) 
connectomes = []
for conn_file in conn_files:
      connectomes += [np.load(conn_file)]

from graph_construction import make_group_graph

graph = make_group_graph(connectomes, self_loops=False, k=8, symmetric=True)
```

## Running model
__*Time windows*__

For the GCN model in order to run the model on different sizes of input, we will concatenate bold data of the same stimuli and save it in a single file.

It means that we need to extract the fmri time-series for each trial using the event design labels.

Different lengths for our input data can be selected. 
In this example we will continue with __*window_length = 1*__, which means each input file will have a length equal to just one Repetition Time (TR).

TR is cycle time between corresponding points in fMRI.

```{code-cell} ipython3
from gcn_model import GCN

window_length = 1
gcn = GCN(graph.edge_index, graph.edge_attr, n_timepoints=window_length)
gcn
```

## Train and evaluating the model

We will repeat the process for 15 epochs (times), and will evaluate the model based on the average accuracy and loss of these epochs.

```{code-cell} ipython3
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)    

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
        print(f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")

def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct
```

```{code-cell} ipython3
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4, weight_decay=5e-4)

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_loop(train_generator, gcn, loss_fn, optimizer)
    loss, correct = valid_test_loop(valid_generator, gcn, loss_fn)
    print(f"Valid metrics:\n\t avg_loss: {loss:>8f};\t avg_accuracy: {(100*correct):>0.1f}%")
```

```{code-cell} ipython3
# results
loss, correct = valid_test_loop(test_generator, gcn, loss_fn) 
print(f"Test metrics:\n\t avg_loss: {loss:>f};\t avg_accuracy: {(100*correct):>0.1f}%")
```
