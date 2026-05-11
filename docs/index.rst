Denoising AutoEncoder - Documentation
=======================================

A PyTorch implementation of a **Denoising Autoencoder (DAE)** for
single-channel speech enhancement, following the approach of Lu et al. (2013)
and with architecture choices informed by Nossier et al. (2020).

The model learns a direct mapping from *noisy* spectral features to *clean*
spectral features. At inference time only the noisy side is available: the
encoder compresses it into a bottleneck, and the decoder reconstructs a clean
estimate from that bottleneck.

.. code-block:: text

   Noisy speech frame                    Clean speech estimate
           │                                      ▲
           ▼                                      │
    ┌─────────────┐    bottleneck z    ┌──────────────┐
    │   Encoder   │ ─────────────────► │    Decoder   │
    └─────────────┘                    └──────────────┘

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   guide/overview
   guide/installation
   guide/quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/nets
   api/data
   api/train
   api/showcases
   api/metrics

References
----------

* Lu, X., Tsao, Y., Matsuda, S., & Hori, C. (2013). *Speech Enhancement Based
  on Deep Denoising Autoencoder*. INTERSPEECH 2013.
* Nossier, Soha A., et al. *An experimental analysis of deep learning 
  architectures for supervised speech enhancement.* _Electronics 10.1_ (2020): 17.
* Thiemann, Joachim, Nobutaka Ito, and Emmanuel Vincent. *The diverse 
  environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings.* _Proceedings of Meetings on Acoustics. Vol. 19. No. 1. Acoustical Society of America_, 2013.
