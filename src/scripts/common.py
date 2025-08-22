#!/usr/bin/env python
"""Common objects in the lfric_hj_bench_code project."""

from dataclasses import dataclass, field

PROJECT = "lfric_egp_bench"


@dataclass
class Group:
    """Details for a group of simulations."""

    title: str
    simulations: tuple
    kw_plt: dict = field(default_factory=dict)


GROUPS = {
    "tf": Group(
        title="Temperature Forcing",
        simulations=(
            "shj",
            "dhj",
            "camembert_case1_gj1214b",
            "camembert_case1_k2-18b",
        ),
    ),
    "gr": Group(
        title="Double-Grey Radiative Transfer",
        simulations=(
            "camembert_case2_gj1214b",
            "camembert_case2_k2-18b",
        ),
    ),
    "rt": Group(
        title="Multiband Radiative Transfer",
        simulations=(
            "camembert_case3_gj1214b",
            "camembert_case3_k2-18b",
        ),
    ),
}


@dataclass
class Experiment:
    """Experiment details."""

    title: str
    planet: tuple
    group: str


EXPERIMENTS = {
    "shj": Experiment(
        title="Shallow Hot Jupiter",
        planet="shj",
        group="tf",
    ),
    "dhj": Experiment(
        title="Deep Hot Jupiter",
        planet="dhj",
        group="tf",
    ),
    "camembert_case1_gj1214b": Experiment(
        title="CAMEMBERT - Case 1 - GJ 1214b",
        planet="camembert_gj1214b",
        group="tf",
    ),
    "camembert_case1_k2-18b": Experiment(
        title="CAMEMBERT - Case 1 - K2-18b",
        planet="camembert_k2-18b",
        group="tf",
    ),
    "camembert_case2_gj1214b": Experiment(
        title="CAMEMBERT - Case 2 - GJ 1214b",
        planet="camembert_gj1214b",
        group="gr",
    ),
    "camembert_case2_k2-18b": Experiment(
        title="CAMEMBERT - Case 2 - K2-18b",
        planet="camembert_k2-18b",
        group="gr",
    ),
    "camembert_case3_gj1214b": Experiment(
        title="CAMEMBERT - Case 3 - GJ 1214b",
        planet="camembert_gj1214b",
        group="rt",
    ),
    "camembert_case3_k2-18b": Experiment(
        title="CAMEMBERT - Case 3 - K2-18b",
        planet="camembert_k2-18b",
        group="rt",
    ),
}
