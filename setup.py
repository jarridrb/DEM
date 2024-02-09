#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dem",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = dem.train:main",
            "eval_command = dem.eval:main",
        ]
    },
)
