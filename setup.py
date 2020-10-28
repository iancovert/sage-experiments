import setuptools

setuptools.setup(
    name="sage-experiments",
    version="0.0.1",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="For calculating global feature importance using Shapley values.",
    long_description="""
        For replicating experiments from [this paper](https://arxiv.org/abs/2004.00668)
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/iancovert/sage-experiments/",
    packages=['sage'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pandas',
        'catboost',
        'xgboost',
        'scikit-learn',
        'torch',
        'torchvision'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
