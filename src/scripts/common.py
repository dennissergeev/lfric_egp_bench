"""Common objects in the lfric_hj_bench_code project."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import paths
from aeolus.const import init_const
from aeolus.const.const import ConstContainer
from aeolus.coord import ensure_bounds
from aeolus.lfric import add_equally_spaced_height_coord, fix_time_coord
from aeolus.model import lfric, um
from aeolus.model.base import Model
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError as CoNotFound
from iris.util import new_axis, promote_aux_coord_to_dim_coord
from spharm import getspecindx
from windspharm.standard import VectorWind
from windspharm.tools import order_latdim, prep_data

PROJECT = "lfric_egp_bench"
TIME_ORIGIN = "2000-01-01 00:00:00"
EARTH_RADIUS = 6_371_200.0  # metres

um.dt_force = "m01s53i181"  # type: ignore
lfric.du_force = "eastward_wind_increment_from_external_forcing"  # type: ignore
um.du_force = "m01s13i385"  # type: ignore
lfric.dv_force = "northward_wind_increment_from_external_forcing"  # type: ignore
um.dv_force = "m01s13i386"  # type: ignore
lfric.dw_force = "vertical_air_velocity_increment_from_external_forcing"  # type: ignore
um.dw_force = "none"  # type: ignore


@dataclass
class InternalFlux:
    """Internal flux specification in RT experiments."""

    value: float = 5.670367
    decorr_timescale: float = 100000.0
    wavenumber: int = 20
    method: Literal["uniform", "non_uniform", "random_harmonics"] = "random_harmonics"


@dataclass
class Experiment:
    """Experiment details."""

    title: str
    const: ConstContainer
    group: str
    label: str
    run_length: int
    timestep: float  # seconds
    resolution: str  # e.g. 'C48' or '144x90'
    n_levels: int  # number of layers
    stretch_factor: float = 1.0  # 1 is no stretching
    target_lon: float = 0.0  # focus of stretching
    category: str = ""
    internal_flux: InternalFlux | None = None


SHJ_BASE = Experiment(
    title="Shallow Hot Jupiter",
    const=init_const("shj", directory=paths.const),
    group="shj",
    label="c48_l32",
    run_length=1200,
    timestep=1200,
    resolution="C48",
    n_levels=32,
    category="tf",
    internal_flux=None,
)

DHJ_BASE = Experiment(
    title="Deep Hot Jupiter",
    const=init_const("dhj", directory=paths.const),
    group="dhj",
    label="c48_l66",
    run_length=1200,
    timestep=45,
    resolution="C48",
    n_levels=66,
    category="tf",
    internal_flux=None,
)

HD209_BASE = Experiment(
    title="HD 209458b",
    const=init_const("hd209458b", directory=paths.const),
    group="hd209",
    label="base_c48",
    run_length=1200,
    timestep=30,
    resolution="C48",
    n_levels=66,
    category="rt",
    internal_flux=InternalFlux(),
)


EXPERIMENTS = {
    # SHJ
    "shj_c48_l32": replace(SHJ_BASE),
    "shj_c24_l32": replace(SHJ_BASE, label="c24_l32", resolution="C24", timestep=2400),
    "shj_c96_l32": replace(SHJ_BASE, label="c96_l32", resolution="C96", timestep=600),
    "shj_c48_l16": replace(SHJ_BASE, label="c48_l16", n_levels=16),
    "shj_c48_l64": replace(SHJ_BASE, label="c48_l64", n_levels=64),
    # DHJ
    "dhj_c48_l66": replace(DHJ_BASE),
    "dhj_c24_l66": replace(DHJ_BASE, label="c24_l66", resolution="C24", timestep=75),
    "dhj_c96_l66": replace(DHJ_BASE, label="c96_l66", resolution="C96", timestep=30),
    # hd209
    "hd209_base_c48": replace(HD209_BASE),
    "hd209_uniform_t100k_c48": replace(
        HD209_BASE, label="uniform_t100k_c48"
    ),  # TODO: rename! flux=0
    "hd209_rand_t100k_nf20_tau1e5": replace(
        HD209_BASE,
        label="rand_t100k_nf20_tau1e5",
        internal_flux=InternalFlux(method="random_harmonics"),
    ),
    # old
    # "shj_base_c48": replace(SHJ_BASE),
    # "dhj_base_c24": replace(DHJ_BASE, timestep=75, resolution="C24"),
    # "dhj_base_c48": replace(DHJ_BASE),
    # "dhj_base_c96": replace(DHJ_BASE, resolution="C96"),
    # "dhj_base_c192": replace(DHJ_BASE, timestep=15, resolution="C192"),
    # "dhj_base_c24_s0p5_lon00": replace(
    #     DHJ_BASE, timestep=75, resolution="C24", stretch_factor=0.5, target_lon=0.0
    # ),
    # "dhj_base_c24_s0p5_lon90": replace(
    #     DHJ_BASE, timestep=75, resolution="C24", stretch_factor=0.5, target_lon=90.0
    # ),
}


@dataclass
class BenchModel:
    """Model details."""

    title: str
    kw_plt: dict = field(default_factory=dict)
    model: Model = lfric
    details: str = ""
    datetime_start: datetime = datetime.strptime(TIME_ORIGIN, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007, RUF009


MODELS = {
    "lfric": BenchModel(
        model=lfric,
        title="LFRic",
        kw_plt={
            "linestyle": "-",
            "linewidth": 1.25,
        },
    ),
    "um": BenchModel(
        model=um,
        title="UM",
        kw_plt={
            "linestyle": "--",
            "linewidth": 0.75,
            "dash_capstyle": "round",
        },
    ),
}


@dataclass
class Category:
    """Details for a category of simulations."""

    title: str
    simulations: Iterable
    kw_plt: dict = field(default_factory=dict)


CATEGORIES = {
    "tf": Category(
        title="Temperature Forcing",
        simulations=(k for k in EXPERIMENTS if k.startswith(("shj", "dhj"))),
    ),
    "rt": Category(
        title="Radiative Transfer",
        simulations=(k for k in EXPERIMENTS if k.startswith("hd209")),
    ),
}


def camembert_days_to_save(exp_label):
    """Give a list of days for the CAMEMBERT output."""
    every_50 = [*range(0, EXPERIMENTS[exp_label].run_length + 1, 50)]
    return (every_50[0:-21:20] + every_50[-21:])[1:]


@dataclass
class Diag:
    """Diagnostics to be plotted."""

    title: str
    units: str
    recipe: Callable
    method: str = "pcolormesh"
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
    except CoNotFound:
        pass
    if cube.units == "ms-1":
        cube.units = "m s-1"


def add_time_coord(cube, field, filename, time_origin=TIME_ORIGIN):
    """Extract time from filename and add as aux coord to cube."""
    dt = datetime.strptime(Path(filename).stem.split("_")[-1].split("-")[1], "%Y%m%d")  # noqa: DTZ007
    dt_sec = (dt - datetime.strptime(time_origin, "%Y-%m-%d %H:%M:%S")).total_seconds()  # noqa: DTZ007
    time_coord = AuxCoord(
        dt_sec,
        standard_name="time",
        units=f"seconds since {time_origin}",
    )
    cube.add_aux_coord(time_coord)
    cube = new_axis(cube, "time")
    promote_aux_coord_to_dim_coord(cube, "time")


def chain_callbacks(*callbacks):
    """
    Combine multiple Iris load callbacks into a single callback.

    Parameters
    ----------
    *callbacks: callable
        Callback functions, each with the signature ``(cube, field, filename)``,
        as expected by `iris.load*`.

    Returns
    -------
    callable
        A single callback that calls each of the given callbacks in order.
    """

    def combined(cube, field, filename):
        for cb in callbacks:
            cb(cube, field, filename)

    return combined


def ke_spectrum(u_cube, v_cube, gridtype="regular", rsphere=EARTH_RADIUS):
    """
    Compute the kinetic energy spectrum as a function of total wavenumber.

    The horizontal wind is decomposed into spherical harmonics and the
    kinetic energy per total (meridional) wavenumber `n` is computed
    from the spectral vorticity and divergence coefficients, following
    Boer (1983) and Koshyk & Hamilton (2001). Summing the returned
    spectrum over wavenumber recovers the global-mean kinetic energy
    of the horizontal wind.

    Parameters
    ----------
    u_cube, v_cube: iris.cube.Cube
        Eastward and northward wind components on a common lat-lon grid,
        with dimensions such as (level_height, latitude, longitude).
        Latitude and longitude must each be a single dimension of the
        cubes; any other dimensions (e.g. level_height, time) are
        preserved in the output.
    gridtype: str, optional
        Either "regular" (default) for an evenly-spaced latitude grid,
        or "gaussian".
    rsphere: float, optional
        Planetary radius in metres, used to scale the spectrum. Defaults
        to Earth's radius; for the exoplanet experiments in this project
        pass e.g. ``const.radius.data.item()`` from the relevant
        `Experiment.const`.

    Returns
    -------
    iris.cube.Cube
        Kinetic energy spectrum in m2 s-2, with a "wavenumber" dimension
        (n = 1, 2, ...) replacing latitude and longitude.
    """
    lat_coord = u_cube.coord(axis="Y")
    lon_coord = u_cube.coord(axis="X")
    (lat_dim,) = u_cube.coord_dims(lat_coord)
    (lon_dim,) = u_cube.coord_dims(lon_coord)

    dim_chars = []
    for dim in range(u_cube.ndim):
        if dim == lat_dim:
            dim_chars.append("y")
        elif dim == lon_dim:
            dim_chars.append("x")
        else:
            dim_chars.append(chr(ord("a") + dim))
    dimorder = "".join(dim_chars)

    u_prep, info = prep_data(u_cube.data, dimorder)
    v_prep, _ = prep_data(v_cube.data, dimorder)
    _, u_prep, v_prep = order_latdim(lat_coord.points, u_prep, v_prep)

    vw = VectorWind(u_prep, v_prep, gridtype=gridtype, rsphere=rsphere)
    ntrunc = vw.s.nlat - 1
    vrtspec, divspec = vw.s.getvrtdivspec(u_prep, v_prep)
    indxm, indxn = getspecindx(ntrunc)

    # Sum |coeff|^2 over the full range m = -n..n. Only m >= 0 is stored,
    # so contributions from m > 0 are doubled to account for m < 0.
    doubling = np.where(indxm == 0, 1.0, 2.0)[:, np.newaxis]
    power = (np.abs(vrtspec) ** 2 + np.abs(divspec) ** 2) * doubling

    ke_n = np.zeros((ntrunc + 1,) + power.shape[1:])
    np.add.at(ke_n, indxn, power)
    n = np.arange(1, ntrunc + 1)
    ke_n = 0.25 * rsphere**2 * ke_n[1:] / (n * (n + 1))[:, np.newaxis]

    # Move the wavenumber axis (currently first) to last, and restore the
    # shape/order of the non-lat-lon ("other") dimensions.
    other_shape = info["intermediate_shape"][2:]
    ke_n = np.moveaxis(ke_n.reshape((ntrunc,) + other_shape), 0, -1)  # type: ignore

    other_dims = [d for d in range(u_cube.ndim) if d not in (lat_dim, lon_dim)]
    dim_coords_and_dims = []
    for new_dim, orig_dim in enumerate(other_dims):
        for coord in u_cube.coords(dim_coords=True):
            if u_cube.coord_dims(coord) == (orig_dim,):
                dim_coords_and_dims.append((coord.copy(), new_dim))
    wavenumber_coord = DimCoord(n, long_name="wavenumber", units="1")
    dim_coords_and_dims.append((wavenumber_coord, ke_n.ndim - 1))

    return Cube(
        ke_n,
        long_name="kinetic_energy_spectrum",
        units="m2 s-2",
        dim_coords_and_dims=dim_coords_and_dims,
    )
