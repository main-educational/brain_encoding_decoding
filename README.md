# brain_encoding_decoding

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://main-educational.github.io/brain_encoding_decoding/intro.html) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/main-educational/brain_encoding_decoding/HEAD)
[![Docker Hub](https://img.shields.io/docker/pulls/user/repo)]() 
[![GitHub size](https://img.shields.io/github/repo-size/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding/archive/master.zip)
[![GitHub issues](https://img.shields.io/github/issues/main-educational/brain_encoding_decoding?style=plastic)](https://github.com/main-educational/brain_encoding_decoding)
[![GitHub PR](https://img.shields.io/github/issues-pr/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding/pulls)
[![License](https://img.shields.io/github/license/main-educational/brain_encoding_decoding)](https://github.com/main-educational/brain_encoding_decoding)
[![CONP](https://img.shields.io/badge/Supported%20by-%20CONP%2FPCNO-red)](https://conp.ca/)



This is a jupyter book presenting an introduction to brain encoding and decoding using python. It is rendered on [main-educational.github.io/brain_encoding_decoding](https://main-educational.github.io/brain_encoding_decoding/intro.html). See the introduction of the jupyter book for more details, and acknowledgements.

### Build the book

If you want to build the book locally:

- Clone this repository
- Run `pip install -r binder/requirements.txt` (it is recommended to run this command in a virtual environment)
- For a clean build, remove `content/_build/`
- Run `jb build content/`

An html version of the jupyter book will be automatically generated in the folder `content/_build/html/`.

### Hosting the book

The html version of the book is hosted on the `gh-pages` branch of this repo. Navigate to your local build and run,
- `ghp-import -n -p -f content/_build/html`

This will automatically push your build to the `gh-pages` branch. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).
