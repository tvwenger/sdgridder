from setuptools import setup

setup(
    name="sdgridder",
    version="0.1",
    description="sdgridder: Gridder for single dish telescope data",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["sdgridder"],
    install_requires=[
        "argparse",
        "numpy",
        "sparse",
        "astropy",
    ],
    scripts=["sdgridder/sdgridder"],
)
