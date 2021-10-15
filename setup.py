#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = [line.strip() for line in f]

setup(
    name="salina",
    version="1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=reqs,
)
