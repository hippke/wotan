Python Interface
================

This describes the Python interface to Wotan.


Define data for a search
------------------------

.. class:: a.model(t, y, dy)

:t: *(array)* Time series of the data (**in units of days**)
:y: *(array)* Flux series of the data, so that ``1`` is nominal flux (out of transit) and ``0`` is darkness. A transit may be represented by a flux of e.g., ``0.99``
:dy: *(array, optional)* Measurement errors of the data

.. note::

   Text
