"""The replacement tree.

`system_rearchitecture.md` §4.1: the new packages are written fresh against the
layering in `checks/layer_map.py`, never by moving a legacy file into them.
Legacy ``engine/`` runs the board unchanged for the whole migration and is
deleted whole at phase 8, at which point this package is renamed to ``engine``.

This module is a namespace container: it holds subpackages and no names.
Importing ``engine.v2`` itself is a violation caught by
``checks/import_layers.py`` — import the subpackage that owns the name.
"""
