"""The reusable domain: generation, scenarios, valuation, simulation.

Four peers on two layers. §4.1 splits ``simulation`` (4b) below the other three
(4a) because a direction check cannot enforce a prohibition between peers: the
scenario builder cannot price a position precisely because valuation is not
beneath it.

A namespace container — it holds subpackages and no names.
"""
