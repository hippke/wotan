Installation
=====================================

Wotan can be installed conveniently using pip::

    pip install wotan

The latest version can be pulled from github::

    git clone https://github.com/hippke/wotan.git
    cd wotan
    python setup.py install

If you don't have ``git`` on your machine, you can find installation instructions `here <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.


Dependencies
------------------------

Wotan requires numpy, scipy and numba to run. Optional:

- `statsmodels` for `lowess`
- `sklearn for `huberspline` and the GPs
- `supersmoother`
- `untrendy`

To install all dependencies, try ``pip install numpy numba scipy statsmodels sklearn supersmoother untrendy``


Compatibility
------------------------

Wotan has been `tested to work <https://travis-ci.com/hippke/wotan>`_ with Python 2.7, 3.5, 3.6, 3.7.
