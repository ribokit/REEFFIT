Installation
------------------

**REEFFIT** can be downloaded for non-commercial use at the `RMDB <https://rmdb.stanford.edu/tools/>`_. Commercial users, please `contact us <https://rmdb.stanford.edu/help/about/#contact>`_.

* Download the zip or tar file of the repository and unpack; or 

.. code-block:: bash

    git clone https://github.com/ribokit/REEFFIT.git

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

* Note that you should have ``RDATKit`` installed and properly set up as well (see `ribokit/RDATKit <https://github.com/ribokit/RDATKit>`_).

* In your profile (e.g. ``.bashrc``), include an environment variable ``REEFFIT_HOME`` that points to the REEFFIT home directory, e.g.:

.. code-block:: bash

    export REEFFIT_HOME=/path/to/REEFFIT

* Be sure to add the ``REEFFIT_HOME/bin`` directory to your path. Similarly in `~/.bashrc` or `~/.bash_profile`, include:

.. code-block:: bash

    export PATH=$PATH:$REEFFIT_HOME/bin

* Check that ``REEFFIT`` is correctly installed. Execute the **REEFFIT** command by running ``reeffit -h`` in your shell.
