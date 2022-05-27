# Introduction
## Brain encoding and decoding
```{figure} haxby_data/brain-encoding-decoding.png
---
width: 800px
name: brain-encoding-decoding-fig
---
To test the consistency of representations in artificial neural networks (ANNs) and the brain, it is possible to **encode** brain activity based on ANN presented with similar stimuli, or **decode** brain activity by predicting the expected ANN activity and corresponding annotation of cognitive states. Figure from [Schrimpf et al. (2020)](https://doi.org/10.1101/407007) {cite:p}`Schrimpf2020-mc`, under a [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
```
This jupyter book presents an introduction to brain encoding and decoding using fMRI. Brain decoding is a type of model where we try to guess what a subject is doing, based on recordings of brain activity. Brain encoding is the reverse operation, where we use machine learning tools to predict the activity of the brain, either based on annotations of the cognitive states of the subject, or using features learned by an artificial neural network presented with the same stimuli as the subject. The tutorials make heavy use of [nilearn](https://nilearn.github.io/stable/index.html)
manipulate and process fMRI data, as well as [scikit-learn](https://scikit-learn.org/stable/) and [pytorch](https://pytorch.org/)
to apply machine learning techniques on the data.

This resource was developed for use at the [Montreal AI and Neuroscience (MAIN)](https://www.main2021.org/)
conference in November 2021.

## Setup

There are two ways to run the tutorials: local installation and Binder.

### Local installation (Recommended)

```{admonition} Install python
:class: tip
:name: python-install-tip
You need to have access to a terminal with Python 3.
If you have setup your environment based on instructions of [MAIN educational installation guide](https://main-educational.github.io/installation.html), you are good to go 🎉

If it not already the case,
[here](https://realpython.com/installing-python/#how-to-check-your-python-version-on-windows)
is a quick guide to install python 3 on any OS.
```

1. Clone/download this repository to your machine and navigate to the directory.

    ```bash
    git clone https://github.com/main-educational/brain_encoding_decoding.git
    cd brain_encoding_decoding
    ```

2. We encourage you to use a virtual environment for this tutorial
    (and for all your projects, that's a good practice).
    To do this, run the following command in your terminal, it will create the
    environment in a folder named `env_tuto`:

    ```bash
    python3 -m venv env_tuto
    ```
    Then the following command will activate the environment:

    ```bash
    source env_tuto/bin/activate
    ```

    Finally, you can install the required libraries:

    ```bash
    pip install -r binder/requirements.txt
    ```

3. Navigate to the content of the book:
    ```bash
    cd content/
    ```

    Now that you are all set, you can run the notebooks with the command:

    ```bash
    jupyter notebook
    ```
    Click on the `.md` files. They will be rendered as jupyter notebooks 🎉

### Binder

If you wish to run the tutorial in Binder, click on the rocket icon to launch the notebook 🚀

```{warning}
The computing resource on Binder is limited.
Some cells might not execute correctly, or the data download will not be completed.
For the full experience, we recommend using the local set up instruction.
```

## Acknowledgements
Parts of the tutorial are directly adapted from a nilearn [tutorial](https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html) on the Haxby dataset.

This tutorial was prepared and presented by
[Pravish Sainath](https://github.com/pravishsainath)
[Shima Rastegarnia](https://github.com/srastegarnia),
[Hao-Ting Wang](https://github.com/htwangtw)
[Loic Tetrel](https://github.com/ltetrel) and [Pierre Bellec](https://github.com/pbellec).

Some images and code are used from a previous iteration of this tutorial, prepared by Dr [Yu Zhang](https://github.com/zhangyu2ustc).

It is rendered here using [Jupyter Book](https://github.com/jupyter/jupyter-book),
<!-- with compute infrastructure provided by the [Canadian Open Neuroscience Platform (CONP)](http://conp.ca). -->

## References

```{bibliography}
:filter: docname in docnames
```
