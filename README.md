# REEFFIT (RNA Ensemble Extraction From Footprinting Insights Technique)

**REEFFIT** is a method to fit RNA secondary structure ensembles to multi-dimensional chemical mapping data. Currently, the method can take data from one and multiple mutate-and-map or mutate-bind-and-map (mutate-and-map plus ligand titrations, e.g. for riboswitches) experiments. Output is a set of weights and expected reactivities for each structure that linearly combine to form the data. If the structural ensemble is not provided, REEFFIT uses RNAstructure to generate a suboptimal set of structures of all mutants.

## Installation

To install **REEFFIT**, simply:

* Run:
```bash
cd path/to/REEFFIT/repo
python setup.py install
```

For system-wide installation, you must have permissions and use with `sudo`.

**REEFFIT** requires the following *Python* packages as dependencies, all of which can be installed through [`pip`](https://pip.pypa.io/).
```json
cvxopt >= 1.1.6
joblib >= 0.5.4
matplotlib >= 1.1.1
numpy >= 1.6.1
scipy >= 0.9.0
pymc >= 2.2

rdatkit >= 1.0.4
```

* Note that you should have `RDATKit` installed and properly set up as well (see https://github.com/ribokit/RDATKit)

* In your profile (e.g. `~/.bashrc`), include an environment variable `REEFFIT_HOME` that points to the **REEFFIT** home directory, e.g.:
```bash
export REEFFIT_HOME=path/to/REEFFIT/repo
```

* Be sure to add `REEFFIT_HOME/bin` to your `PATH`. Similarly in `~/.bashrc` or `~/.bash_profile`, include:
```bash
export PATH=$PATH:$REEFFIT_HOME/bin
```

* Check that **REEFFIT** is correctly installed. Test the `reeffit` command by running `reeffit -h` in your shell.

## Documentation

Documentation is available at https://reeffit.readthedocs.org/ or https://ribokit.github.io/REEFFIT/.

## License

Copyright &copy; of **REEFFIT** _Source Code_ is described in [LICENSE.md](https://github.com/ribokit/REEFFIT/blob/master/LICENSE.md).

## Reference

> Cordero, P., and Das, R. (**2015**)<br/>
>[Rich structure landscapes in both natural and artificial RNAs revealed by mutate-and-map analysis.](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004473)<br/>
>*PLOS Computational Biology* **11 (11)**: e1004473.

<hr/>

Developed by **Das lab**, _Leland Stanford Junior University_.

README by [**t47**](https://t47.io/), *May 2016*.

