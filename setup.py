from setuptools import setup

setup(name='REEFFIT',
        version='0.5',
        description='RNA Ensemble Extraction From Footprinting Insights Technique',
        author='Pablo Cordero',
        author_email='tsuname@stanford.edu',
        url='http://rmdb.stanford.edu/reeffit',
        packages=['reeffit'],
        install_requires=['numpy', 'scipy', 'joblib', 'pymc', 'matplotlib', 'cvxopt']
        )
