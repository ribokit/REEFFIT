Installation
------------------

**REEFFIT** can be downloaded for non-commercial use at the `RMDB <https://rmdb.stanford.edu/tools/>`_. Commercial users, please `contact us <https://rmdb.stanford.edu/help/about/#contact>`_.

* Download the zip or tar file of the repository and unpack; or 

.. code-block:: bash

    git clone https://github.com/DasLab/REEFFIT.git

* To install ``REEFFIT``, simply:

.. code-block:: bash

    cd path/to/REEFFIT/
    python setup.py install

For system-wide installation, you must have permissions and use with ``sudo``.

``REEFFIT`` requires the following *Python* packages as dependencies, all of which can be installed through `pip <https://pip.pypa.io/>`_:

.. code-block:: js

    cvxopt >= 1.1.6
    joblib >= 0.5.4
    matplotlib >= 1.1.1
    numpy >= 1.6.1
    scipy >= 0.9.0
    pymc >= 2.2

    rdatkit >= 1.0.4

* In your profile (e.g. ``.bashrc)``, include an environment variable ``REEFFIT_HOME`` that points to the REEFFIT home directory. Also add the ``REEFFIT_HOME/bin`` directory to your path:

.. code-block:: bash

    export REEFFIT_HOME=/path/to/REEFFIT
    export PATH=$PATH:$REEFFIT_HOME/bin

* Check that ``REEFFIT`` is correctly installed. Execute the reeffit command by running ``reeffit -h`` in your shell.
