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
 In the field of functional magnetic resonance imaging (fMRI), one of the first studies which have demonstrated the feasibility of brain decoding was the study by Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. In this study, subjects were presented with various images drawn from different categories. In this example, we try to decode the category of the image presented to the subject from brain data. We are first going to use nilearn to download one subject (number 4) from the Haxby dataset.

```{code-cell} python3
:tags: ["remove-output"]
import os
from nilearn import datasets
# We are fetching the data for subject 4
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_ds = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_ds.func[0]
```

## References

```{bibliography}
:filter: docname in docnames
```
