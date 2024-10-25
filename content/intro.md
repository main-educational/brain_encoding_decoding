<!-- #region -->
# Welcome 

**"Introduction to brain decoding in fMRI"**

This `jupyter book` presents an introduction to `brain decoding` using `fMRI`. It was developed within the [educational courses](https://main-educational.github.io), conducted as part of the [Montreal AI and Neuroscience (MAIN) conference](https://www.main2024.org/) in October 2024.

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://main-educational.github.io/brain_encoding_decoding/intro.html) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/main-educational/brain_encoding_decoding/HEAD)
[![Docker Hub](https://img.shields.io/docker/pulls/user/repo)]() 
[![GitHub size](https://img.shields.io/github/repo-size/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding/archive/master.zip)
[![GitHub issues](https://img.shields.io/github/issues/main-educational/brain_encoding_decoding?style=plastic)](https://github.com/main-educational/brain_encoding_decoding)
[![GitHub PR](https://img.shields.io/github/issues-pr/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding/pulls)
[![License](https://img.shields.io/github/license/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding)
[![CONP](https://img.shields.io/badge/Supported%20by-%20CONP%2FPCNO-red)](https://conp.ca/)

Building upon the prior sections of the [educational courses](https://main-educational.github.io), the here presented resources aim to provide an overview of how `decoding models` can be applied to `fMRI` data in order to investigate `brain function`. Importantly, the respective methods cannot only be utilized to analyze data from `biological agents` (e.g. `humans`, `non-human primates`, etc.) but also `artificial neural networks`, as well as presenting the opportunity to compare processing in both. They are thus core approaches that are prominently used at the intersection of `neuroscience` and `AI`.
 
 
```{figure} haxby_data/brain-encoding-decoding.png
---
width: 800px
name: brain-encoding-decoding-fig
---

To test the consistency of representations in artificial neural networks (ANNs) and the brain, it is possible to **encode** brain activity based on ANN presented with similar stimuli, or **decode** brain activity by predicting the expected ANN activity and corresponding annotation of cognitive states. Figure from [Schrimpf et al. (2020)](https://doi.org/10.1101/407007) {cite:p}`Schrimpf2020-mc`, under a [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
``` 

The tutorials make heavy use of [nilearn](https://nilearn.github.io/stable/index.html) concerning 
manipulating and processing `fMRI` data, as well as [scikit-learn](https://scikit-learn.org/stable/) and [pytorch](https://pytorch.org/) to apply `decoding models` on the data.

We used the [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) framework to provide all materials in an open, structured and interactive manner. ALl pages and section you see here are built from `markdown` files or `jupyter notebooks`, allowing you to read through the materials and/or run them, locally or in the cloud. The three symbols on the top right allow to enable full screen mode, link to the underlying [GitHub repository](https://github.com/main-educational/brain_encoding_decoding) and allow you to download the respective sections as a `pdf` or `jupyter notebook` respectively. Some sections will additionally have a little rocket in that row which will allow you to interactively rerun certain parts via cloud computing (please see the [Binder](#Binder) section for more information).


## Brain decoding vs. encoding

In short, `encoding` and `decoding` entail contrary operations that can yet be utilized in a complementary manner. `Encoding models` applied to `brain data`, e.g. `fMRI`, aim to predict `brain responses`/`activity` based on `annotations` or `features` of the `stimuli` perceived by the `participant`. These can be obtained from a multitude of options, including `artificial neural networks` which would allow to relate their `processing` of the `stimuli` to that of `biological agents`, ie `brains`. 
`Decoding models` on the other hand comprise `models` with which we aim to `estimate`/`predict` what a `participant` is `perceiving` or `doing` based on `recordings` of `brain responses`/`activity`, e.g. `fMRI`. 

```{figure} graphics/brain_encoding_decoding_example.png
---
width: 800px
name: brain_encoding_decoding_example_fig
---

`Encoding` and `decoding` present contrary, yet complementary operations. While the former targets the prediction of `brain activity`/`responses` based on stimulus percepts/features (e.g. vision & audition), cognitive states or behavior, the latter aims to predict those aspects based on `brain activity`/`responses`. 
``` 

More information and their application can be found in the respective sections of this resource. You can either use the `ToC` on the left or the links below to navigate accordingly.


::::{card-carousel} 2

:::{card}
:margin: 3
:class-body: text-center
:class-header: bg-light text-center
:link: https://main-educational.github.io/brain_encoding_decoding/haxby_data.html
**An overview of the Haxby Dataset**
^^^
```{image} https://main-educational.github.io/brain_encoding_decoding/_images/d3731383fc66953ff04a680a4d6671e6cfbaa19d6fda5f0089239e37c384ac71.png
:height: 100
```

Explore and prepare the tutorial dataset.
+++
Explore this tutorial {fas}`arrow-right`
:::

:::{card}
:margin: 3
:class-body: text-center
:class-header: bg-light text-center
:link: https://main-educational.github.io/brain_encoding_decoding/svm_decoding.html

**Brain decoding with SVM**
^^^
```{image} https://main-educational.github.io/brain_encoding_decoding/_images/2021c085709559df545bf08eb2ee051f9098c2f5619e666dceeb879ff1801dfb.png
:height: 100
```

Utilizing an SVM classifier to predict percepts from fMRI data.
+++
Explore this tutorial {fas}`arrow-right`
:::
::::

::::{card-carousel} 2

:::{card}
:margin: 3
:class-body: text-center
:class-header: bg-light text-center
:link: https://main-educational.github.io/brain_encoding_decoding/mlp_decoding.html

**Brain decoding with MLP**
^^^
```{image} https://main-educational.github.io/brain_encoding_decoding/_images/multilayer-perceptron.png
:height: 100
```

Brain decoding using a basic artificial neural network.
+++
Explore this tutorial {fas}`arrow-right`
:::

:::{card}
:margin: 3
:class-body: text-center
:class-header: bg-light text-center
:link: https://main-educational.github.io/brain_encoding_decoding/gcn_decoding.html

**Brain decoding with GCN**
^^^
```{image} https://main-educational.github.io/brain_encoding_decoding/_images/GCN_pipeline_main2022.png
:height: 100
```

Graph convolutional networks for brain decoding. 
+++
Explore this tutorial {fas}`arrow-right`
:::
::::


## Setup

There are two ways to run the tutorials: `local installation` and using free cloud computing provided by [Binder](https://mybinder.org/). As noted below, we strongly recommend the `local installation`, as the `Binder` option comes with limited computational resources, as well as the missed possibility to directly further explore the `approaches` presented in this tutorial on your own machine. 


````{tab-set}
```{tab-item}  Local installation (Recommended)

For the `local installation` to work, you need two things: the fitting `python environment` and the `content`. Concerning `python`, please have a look at the hint below.

:::{admonition} Install python
:class: tip
:name: python-install-tip
You need to have access to a `terminal` with `Python 3`.
If you have setup your environment based on instructions of [MAIN educational installation guide](https://main-educational.github.io/installation.html), you are good to go 🎉

If it not already the case,
[here](https://realpython.com/installing-python/#how-to-check-your-python-version-on-windows)
is a quick guide to install python 3 on any OS.
:::

After making sure you have a working `python installation`, you need to get the `content` that is going to presented during the tutorial. In more detail, this is done via interactive `jupyter notebooks` which you can obtain by following the steps below:

1. Clone/download this repository to your machine and navigate to the directory.

    ```bash
    git clone https://github.com/main-educational/brain_encoding_decoding.git
    cd brain_encoding_decoding
    ```

2. We encourage you to use a `virtual environment` for this tutorial
    (and for all your projects, that's a good practice).
    To do this, run the following commands in your terminal, it will create the
    `environment` in a folder named `main_edu_brain_decoding`:

    ```bash
    python3 -m venv main_edu_brain_decoding
    ```
    Then the following `command` will `activate` the `environment`:

    ```bash
    source main_edu_brain_decoding/bin/activate
    ```

    Finally, you can install the required `libraries`:

    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the `content` of the `jupyter book`:
    ```bash
    cd content/
    ```

    Now that you are all set, you can run the notebooks with the command:

    ```bash
    jupyter notebook
    ```
    Click on the `.md` files. They will be rendered as jupyter notebooks 🎉

Alternatively, you can use [conda/miniconda](https://docs.conda.io/projects/conda/en/latest/index.html) to create the needed `python environment` like so:
    
    git clone https://github.com/main-educational/brain_encoding_decoding.git
    cd brain_encoding_decoding
    conda env create -f environment.yml
    

```

```{tab-item} Cloud computing with Binder
If you wish to run the tutorial in `Binder`, click on the rocket icon 🚀 in the top right of a given `notebook` to launch it on `Binder`.

:::{warning}
The computing resource on `Binder` is limited.
Some cells might not execute correctly, or the data download will not be completed.
For the full experience, we recommend using the local set up instruction.
:::

```
````


## Instructors

This tutorial was prepared and presented by

::::{card-carousel} 2

:::{card} Pierre-Louis Barbarant
:margin: 3
:class-body: text-center
:link: https://github.com/pbarbarant
:img-top: https://avatars.githubusercontent.com/u/104081777?v=4
:::

:::{card} Peer Herholz
:margin: 3
:class-body: text-center
:link: https://github.com/PeerHerholz
:img-top: https://avatars.githubusercontent.com/u/20129524?v=4?s=100
:::

::::

It is based on earlier versions created by:

::::{card-carousel} 3
:::{card} Isil Bilgin
:margin: 3
:class-body: text-center
:link: https://github.com/complexbrains
:img-top: https://avatars.githubusercontent.com/u/45263281?v=4
:::

:::{card} Alexandre Pasquiou
:margin: 3
:class-body: text-center
:link: https://twitter.com/a_pasquiou
:img-top: https://pbs.twimg.com/profile_images/1542505896386764800/pyC2rgHp_400x400.jpg
:::

:::{card} Pravish Sainath
:margin: 3
:class-body: text-center
:link: https://github.com/pravishsainath
:img-top: https://avatars.githubusercontent.com/u/13696562?v=4
:::
::::

## Thanks and acknowledgements 
Parts of the tutorial are directly adapted from a [nilearn tutorial](https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html) on the so-called [Haxby dataset]().

It was adapted from a prior version which was prepared and presented by
[Pravish Sainath](https://github.com/pravishsainath)
[Shima Rastegarnia](https://github.com/srastegarnia),
[Hao-Ting Wang](https://github.com/htwangtw)
[Loic Tetrel](https://github.com/ltetrel) and [Pierre Bellec](https://github.com/pbellec).

Furthermore, some `images` and `code` are used from a previous iteration of this `tutorial`, prepared by [Dr Yu Zhang](https://github.com/zhangyu2ustc).

We would like to thank the Jupyter community, specifically, the Executable/Jupyter Book and mybinder project for enabling us to create this tutorial. Furthermore, we are grateful for the entire open neuroscience community and the amazing support and resources it provides. This includes the community driven development of data and processing standards, as well as unbelievable amount of software packages that make the here introduced approaches possible to begin with. 

The tutorial is rendered here using [Jupyter Book](https://github.com/jupyter/jupyter-book).

## References

```{bibliography}
:filter: docname in docnames
```
---
<!-- #endregion -->
