``REEFFIT`` Documentation
===================================

``REEFFIT`` (RNA Ensemble Extraction From Footprinting Instights Tool) is a method to fit RNA secondary structure ensembles to multi-dimensional chemical mapping (MCM) data. Currently, the method can take data from multiple mutate-and-map or mutate-bind-and-map (mutate-and-map plus ligand titrations, e.g. for riboswitches) experiments. Output is a set of weights and expected reactivities for each structure that linearly combine to form the data. If the structural ensemble is not provided, ``REEFFIT`` uses ``RNAstructure`` to generate a suboptimal set of structures of all mutants.

----------

Table of Contents
----------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   technical
   install
   run
   license

----------

Reference
----------------------------

| Cordero, P., and Das, R. (**2015**)
| `Rich Structure Landscapes in both Natural and Artificial RNAs Revealed by Mutate-and-Map Analysis. <http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004473>`_
| *PLOS Computational Biology* **11 (11)**: e1004473.

----------

Related Packages
----------------------------

* `HiTRACE </HiTRACE/>`_
* `RDATKit </RDATKit/>`_

----------

Workflows
----------------------------

* `I think my RNA has interesting alternative states </workflows/alternative_states/>`_

----------


Developed by **Das lab**, `Leland Stanford Junior University`.

README by `t47 <https://t47.io/>`_, *May 2016*.
