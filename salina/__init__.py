#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from .agent import Agent, TAgent
from .workspace import Workspace

trace_workspace = False
trace = []
trace_maximum_size = 10000


def instantiate_class(arguments):
    from importlib import import_module

    d = dict(arguments)
    classname = d["classname"]
    del d["classname"]
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c(**d)


def get_class(arguments):
    from importlib import import_module

    if isinstance(arguments, dict):
        classname = arguments["classname"]
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c
    else:
        classname = arguments.classname
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c


def get_arguments(arguments):
    from importlib import import_module

    d = dict(arguments)
    if "classname" in d:
        del d["classname"]
    return d
