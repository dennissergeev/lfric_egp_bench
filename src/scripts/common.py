#!/usr/bin/env python
"""Common objects in the lfric_hj_bench_code project."""

from dataclasses import dataclass, field

import aeolus
import paths
from aeolus.const import init_const
from aeolus.coord import ensure_bounds
from aeolus.lfric import add_equally_spaced_height_coord, fix_time_coord
from aeolus.model import lfric, um
from iris.exceptions import CoordinateNotFoundError as CoNotFound
from iris.util import promote_aux_coord_to_dim_coord

PROJECT = "lfric_egp_bench"


@dataclass
class BenchModel:
    """Model details."""

    title: str
    kw_plt: dict = field(default_factory=dict)
    model: aeolus.model.base.Model = lfric
    timestep: float = 120.0  # seconds


MODELS = {
    "lfric": BenchModel(
        model=lfric,
        title="LFRic",
        timestep=1200.0,
        kw_plt={
            "linestyle": "-",
            "linewidth": 1.25,
        },
    ),
    "um": BenchModel(
        model=um,
        title="UM",
        timestep=1200.0,
        kw_plt={
            "linestyle": "--",
            "linewidth": 0.75,
            "dash_capstyle": "round",
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
    const: aeolus.const.const.ConstContainer
    group: str


EXPERIMENTS = {
    "shj": Experiment(
        title="Shallow Hot Jupiter",
        const=init_const("shj", directory=paths.const),
        group="tf",
    ),
    "dhj": Experiment(
        title="Deep Hot Jupiter",
        const=init_const("dhj", directory=paths.const),
        group="tf",
    ),
    "camembert_case1_gj1214b": Experiment(
        title="CAMEMBERT - Case 1 - GJ 1214b",
        const=init_const("camembert_gj1214b", directory=paths.const),
        group="tf",
    ),
    "camembert_case1_k2-18b": Experiment(
        title="CAMEMBERT - Case 1 - K2-18b",
        const=init_const("camembert_k2-18b", directory=paths.const),
        group="tf",
    ),
    "camembert_case2_gj1214b": Experiment(
        title="CAMEMBERT - Case 2 - GJ 1214b",
        const=init_const("camembert_gj1214b", directory=paths.const),
        group="gr",
    ),
    "camembert_case2_k2-18b": Experiment(
        title="CAMEMBERT - Case 2 - K2-18b",
        const=init_const("camembert_k2-18b", directory=paths.const),
        group="gr",
    ),
    "camembert_case3_gj1214b": Experiment(
        title="CAMEMBERT - Case 3 - GJ 1214b",
        const=init_const("camembert_gj1214b", directory=paths.const),
        group="rt",
    ),
    "camembert_case3_k2-18b": Experiment(
        title="CAMEMBERT - Case 3 - K2-18b",
        const=init_const("camembert_k2-18b", directory=paths.const),
        group="rt",
    ),
}


@dataclass
class Diag:
    """Diagnostics to be plotted."""

    title: str
    units: str
    recipe: callable
    cnorm: bool = False
    kw_plt: dict = field(default_factory=dict)


def replace_z_coord(cube, remove_level_coord=False, inplace=False, model=lfric):
    """
    Replace levels coordinate with level_height.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube.
    model: aeolus.model.Model, optional
        Model class with relevant coordinate names.

    Returns
    -------
    iris.cube.Cube
        Copy of the input cube with a new vertical coordinate.
    """
    if inplace:
        new_cube = cube
    else:
        new_cube = cube.copy()
    promote_aux_coord_to_dim_coord(new_cube, model.z)
    # Reset bounds for the new z coordinate
    new_cube.coord(model.z).bounds = None
    ensure_bounds(new_cube, coords=[model.z])
    # Remove the old levels coordinate
    if remove_level_coord:
        lev_coords = [
            i.name()
            for i in new_cube.coords()
            if i.name() in ["full_levels", "half_levels", "model_level_number"]
        ]
        for coord in lev_coords:
            try:
                new_cube.remove_coord(coord)
            except CoNotFound:
                pass
    if not inplace:
        return new_cube


def lfric_callback_uniform_height(cube, field, filename, model_top_height):
    """Callback function to apply multiple transformations."""
    fix_time_coord(cube, field, filename)
    add_equally_spaced_height_coord(
        cube, field, filename, model_top_height=model_top_height
    )
    try:
        replace_z_coord(cube, inplace=True, remove_level_coord=True, model=lfric)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    if cube.units == "ms-1":
        cube.units = "m s-1"
