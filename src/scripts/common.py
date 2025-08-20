#!/usr/bin/env python
"""Common objects in the lfric_hj_bench_code project."""

from dataclasses import dataclass, field

# Local modules
import paths

# External modules
from aeolus.model import lfric, um


@dataclass
class BenchModel:
    """Simulation details."""

    title: str
    planet: tuple
    timestep: int
    c_num: int
    vert_lev: str
    kw_plt: dict = field(default_factory=dict)


MODELS = {
    "um": BenchModel(
        model=um,
        title="UM",
        data_proc=paths.results_proc_um,
        data_raw=paths.results_raw_um,
        kw_plt={
            "linestyle": "--",
            "linewidth": 0.75,
            "dash_capstyle": "round",
        },
    ),
    "lfric": BenchModel(
        model=lfric,
        title="LFRic-Atmosphere",
        data_proc=paths.results_proc_lfric,
        data_raw=paths.results_raw_lfric,
        kw_plt={
            "linestyle": "-",
            "linewidth": 1.25,
        },
    ),
}


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
        title="Grey Radiative Transfer",
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
class Simulation:
    """LFRic simulation details."""

    title: str
    planet: tuple
    timestep: int
    time_mean_period: int
    resolution: int
    vert_lev: str
    proc_fname_suffix: str
    group: str
    kw_plt: dict = field(default_factory=dict)


SIMULATIONS = {
    "shj": Simulation(
        title="Shallow Hot Jupiter",
        planet="shj",
        resolution="C24",
        kw_plt={"color": "C0"},
        timestep=1200,
        time_mean_period=1000,
        proc_fname_suffix="sigma_p",
        group="tf",
    ),
    "dhj": Simulation(
        title="Deep Hot Jupiter",
        planet="dhj",
        resolution="C24",
        kw_plt={"color": "C1"},
        timestep=120,
        time_mean_period=1000,
        proc_fname_suffix="sigma_p",
        group="tf",
    ),
    "camembert_case3_gj1214b": Simulation(
        title="CAMEMBERT - Case 3 - GJ 1214b",
        planet="camembert_gj1214b",
        resolution="C24",
        kw_plt={"color": "C0"},
        timestep=120,
        time_mean_period=1000,
        proc_fname_suffix="sigma_p",
        group="rt",
    ),
    "camembert_case3_k2-18b": Simulation(
        title="CAMEMBERT - Case 3 - K2-18b",
        planet="camembert_k2-18b",
        resolution="C24",
        kw_plt={"color": "C1"},
        timestep=120,
        time_mean_period=1000,
        proc_fname_suffix="sigma_p",
        group="rt",
    ),
}
