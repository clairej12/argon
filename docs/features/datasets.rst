Data and Datasets
=================

There are two main modules for working with data and datasets respectively:

:py:mod:`argon.data` contains utilities for working with
:py:class:`Data <argon.data.Data>` objects
containing structured pytrees which can be loaded
from disk via :py:func:`DataLoader <argon.data.DataLoader>`
similar to PyTorch.

:py:mod:`argon.datasets` contains tools for downloading and working 
with common datasets, as well as a :py:class:`DatasetRegistry <argon.datasets.DatasetRegistry>`
which can be used to register new datasets.