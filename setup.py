from setuptools import setup, find_packages

from reeffit.__init__ import __version__

setup(
    name='REEFFIT',
    description='RNA Ensemble Extraction From Footprinting Insights Technique',
    keywords='RNA Ensemble Structure Weight Alternative',
    version=__version__,

    author='Pablo Cordero, Rhiju Das',
    author_email='rhiju@stanford.edu',

    url='https://github.com/ribokit/reeffit/',
    license='https://rmdb.stanford.edu/reeffit',

    packages=find_packages(),
    install_requires=open('requirements.txt', 'r').readlines(),
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7'
    ),

    zip_safe=True
)

