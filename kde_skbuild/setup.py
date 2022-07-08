from skbuild import setup

setup(
    name="kde_skbuild",
    author="Intel Corporation",
    version="0.0.1",
    description="An example of data-parallel Python extensions built with scikit-build and oneAPI DPC++",
    long_description="""
    Example of using oneAPI to build data-parallel extension using setuptools.

    Part of oneAPI for Scientific Python community virtual poster.
    Also see README.md
    """,
    license="Apache 2.0",
    packages=["kde_skbuild",]
)
