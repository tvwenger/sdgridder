from setuptools import setup

setup(
    name="tvwgridder",
    version="0.1",
    description="TVWGridder: Grid GBT SDFITS data",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["tvwgridder"],
    install_requires=[
        "argparse",
        "numpy",
        "sparse",
        "astropy",
    ],
    scripts=["tvwgridder/tvwgridder"],
)
