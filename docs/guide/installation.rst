Installation
============

Requirements
------------

* Python ≥ 3.12
* `uv <https://github.com/astral-sh/uv>`_ (recommended) or pip

Clone and install
-----------------

.. code-block:: bash

    git clone <repo-url>
    cd speech_enhancement

    # With uv (recommended)
    uv pip install -e .

    # Or with pip inside a virtual environment
    pip install -e .

Optional documentation dependencies
-------------------------------------

To build these docs locally, install Sphinx and the Furo theme::

    uv pip install sphinx furo

Then from the ``docs/`` directory::

    make html
    xdg-open _build/html/index.html   # Linux
    open _build/html/index.html        # macOS

Data
----

Download the datasets and place them under ``data/`` in the repository root.

**LibriSpeech** (train + test)::

    data/
    ├── train-clean-100/LibriSpeech/train-clean-100/<speaker>/<chapter>/*.flac
    └── test-clean/LibriSpeech/test-clean/<speaker>/<chapter>/*.flac

Download from https://www.openslr.org/12 (~6.3 GB for train-clean-100).

**DEMAND noise corpus**::

    data/noise/DEMAND/<ENVIRONMENT_16k>/ch01.wav

Download from https://zenodo.org/record/1227121 and extract under
``data/noise/DEMAND/``.
