.. MULTICOM_ligand documentation master file, created by
   sphinx-quickstart on Tue Jan 21 15:49:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MULTICOM_ligand's documentation!
============================================


.. mdinclude:: ../../README.md
    :start-line: 4
    :end-line: 12

.. image:: ./_static/MULTICOM_ligand.png
  :alt: Overview of MULTICOM_ligand
  :align: center
  :width: 600

.. mdinclude:: ../../README.md
    :start-line: 18
    :end-line: 20


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   data_preparation
   available_methods
   method_inference
   ensemble_inference
   comparative_plots
   for_developers
   acknowledgements
   citing_this_work
   bonus

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: Default Configs

   configs/analysis
   configs/data
   configs/model

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   modules/multicom_ligand.binding_site_crop_preparation
   modules/multicom_ligand.complex_alignment
   modules/multicom_ligand.inference_relaxation
   modules/multicom_ligand.minimize_energy
   modules/multicom_ligand.ensemble_generation
   modules/multicom_ligand.data_utils
   modules/multicom_ligand.model_utils
   modules/multicom_ligand.utils
   modules/multicom_ligand.resolvers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
