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

# The Haxby dataset

## Downloading the data
 In the field of functional magnetic resonance imaging (fMRI), one of the first studies which have demonstrated the feasibility of brain decoding was the study by Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. Subjects were presented with various images drawn from different categories. In this tutorial, we will try to decode the category of the image presented to the subject from brain data. We are first going to use nilearn to download one subject (number 4) from the Haxby dataset.

```{code-cell} python3
:tags: ["remove-output"]
import os
from nilearn import datasets
# We are fetching the data for subject 4
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]
```
## Visualizing the data
The data includes `nii` files, which contains images of brain volumes, either anatomical or functional. We can examine one functional volume using nilearn's plotting tools. Because fmri data are 4D we use [nilearn.image.mean_img](https://nilearn.github.io/modules/generated/nilearn.image.mean_img.html#nilearn.image.mean_img) to extract the average brain volume.
```{code-cell} python3
from nilearn import plotting
from nilearn.image import mean_img
plotting.view_img(mean_img(func_file), threshold=None)
```
Note that it is very hard to see the anatomy of the brain with that type of image. But it is not designed to capture brain anatomy, but rather changes of brain activity over time, through the coupling of neuronal activity with the oxygenation of blood vessels.

```{admonition} Interactive viewer
:class: tip
The viewer `plotting.view_img` is interactive. You can click on the brain volume to explore different slices. You can learn about the three anatomical planes: sagittal (`x`), coronal (`y`) and axial (`z`) in the [wikipedia article](https://en.wikipedia.org/wiki/Anatomical_plane).
```

## Preparing the fMRI data
```{figure} haxby_data/masker.png
---
width: 800px
name: masker-fig
---
A `Masker` object is used to convert a 4D volume (space + time) into a data array, where each column is a voxel or brain region (features) and each row is a time point (samples). Figure from the [nilearn documentation](https://nilearn.github.io/stable/manipulating_images/masker_objects.html).
```
Instead of keeping the fMRI data as a 4D array (3D spatial coordinates + time), we are going to extract the time series associated with a mask of the ventral temporal cortex. This mask has been generated as part of the Haxby et al. (2001) study, and highlights a part of the brain specialized in the processing of visual information, and which contains areas sensitive to different types of image categories.

```{code-cell} python3
mask_filename = haxby_dataset.mask_vt[0]
# Let's visualize it, using the subject's anatomical image as a
# background
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                  cmap='Paired')
```

We use one of the nilearn maskers to extract the fMRI time series associated just with this mask:
```{code-cell} python3
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
# Selecting data
X = masker.fit_transform(func_file)
print(X.shape)
```
We can see that the dataset has 1452 time samples (number of rows) and 675 voxels in the mask (number of columns).

```{admonition} Nilearn maskers
:class: tip
Nilearn maskers are very versatile and offer many approaches to extract a time series array from 4D data, as well as reshape back an array into a series of brain volumes. You can learn more about nilearn maskers in this [documentation](https://nilearn.github.io/stable/manipulating_images/masker_objects.html).
```
## References

```{bibliography}
:filter: docname in docnames
```
