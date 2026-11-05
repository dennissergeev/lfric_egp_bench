"""Common objects in the lfric_hj_bench_code project."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal

import iris
import numpy as np
import paths
from aeolus.const import init_const
from aeolus.const.const import ConstContainer
from aeolus.coord import ensure_bounds
from aeolus.lfric import (
    add_equally_spaced_height_coord,
    fix_time_coord,
    load_lfric_raw,
)
from aeolus.model import lfric, um
from aeolus.model.base import Model
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError as CoNotFound
from iris.fileformats.pp import EARTH_RADIUS
from iris.util import new_axis, promote_aux_coord_to_dim_coord

PROJECT = "lfric_egp_bench"
TIME_ORIGIN = "2000-01-01 00:00:00"
PA_TO_BAR = 1e-5
# Diagnostic files written by XIOS on the regular lat-lon grid
LATLON_GLOB = "lfric_diag_latlon*.nc"
# Line style of the reference lines marking the target pressure and mean period
KW_REF_LINE = {"color": "k", "lw": 1, "ls": "--", "dash_capstyle": "round"}
# Global attributes common to all netCDF files written by the project scripts
GLOBAL_ATTRS = {
    "project": PROJECT,
    "creator_name": "Denis E. Sergeev",
    "creator_email": "denis.sergeev@bristol.ac.uk",
    "creator_institution": "University of Bristol",
    "institution": "University of Bristol",
    "source": "LFRic atmosphere model (lfric_atm)",
    "coverage_content_type": "modelResult",
}

# Diagnostics missing from the standard aeolus model container
lfric.w = "w_in_w3"  # type: ignore
lfric.dt_force = "temperature_increment_from_external_forcing"  # type: ignore
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
    heating_amplitude: float = 0.0
    scale_heights: float = 1.0


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
    n_radstep: int | None = None  # number of radiative calls per time step
    stretch_factor: float = 1.0  # 1 is no stretching
    target_lon: float = 0.0  # focus of stretching
    category: str = ""
    internal_flux: InternalFlux | None = None
    kw_plt: dict = field(default_factory=dict)


SHJ_BASE = Experiment(
    title="C48 L32 (control)",
    const=init_const("shj", directory=paths.const),
    group="shj",
    label="c48_l32",
    run_length=1200,
    timestep=1200,
    resolution="C48",
    n_levels=32,
    category="tf",
    internal_flux=None,
    kw_plt={"color": "C0", "linewidth": 1.5},
)

DHJ_BASE = Experiment(
    title="C48 (control)",
    const=init_const("dhj", directory=paths.const),
    group="dhj",
    label="c48_l66",
    run_length=1200,
    timestep=45,
    resolution="C48",
    n_levels=66,
    category="tf",
    internal_flux=None,
    kw_plt={"color": "C0", "linewidth": 1.5},
)

HD209_BASE = Experiment(
    title="HD 209458b",
    const=init_const("hd209458b", directory=paths.const),
    group="hd209",
    label="base_c48",
    run_length=1200,
    timestep=30,
    n_radstep=4,
    resolution="C48",
    n_levels=66,
    category="rt",
    internal_flux=InternalFlux(),
    kw_plt={"color": "C0", "linewidth": 1.5},
)


EXPERIMENTS = {
    # SHJ
    "shj_c48_l32": replace(SHJ_BASE),
    "shj_c24_l32": replace(
        SHJ_BASE,
        title="C24 L32",
        label="c24_l32",
        resolution="C24",
        timestep=2400,
        kw_plt={"color": "C1"},
    ),
    "shj_c96_l32": replace(
        SHJ_BASE,
        title="C96 L32",
        label="c96_l32",
        resolution="C96",
        timestep=600,
        kw_plt={"color": "C2"},
    ),
    "shj_c48_l16": replace(
        SHJ_BASE,
        title="C48 L16",
        label="c48_l16",
        n_levels=16,
        kw_plt={"color": "C3"},
    ),
    "shj_c48_l64": replace(
        SHJ_BASE,
        title="C48 L64",
        label="c48_l64",
        n_levels=64,
        kw_plt={"color": "C4"},
    ),
    # DHJ
    "dhj_c48_l66": replace(DHJ_BASE),
    "dhj_c24_l66": replace(
        DHJ_BASE,
        title="C24",
        label="c24_l66",
        resolution="C24",
        timestep=75,
        kw_plt={"color": "C1"},
    ),
    "dhj_c96_l66": replace(
        DHJ_BASE,
        title="C96",
        label="c96_l66",
        resolution="C96",
        timestep=30,
        kw_plt={"color": "C2"},
    ),
    "dhj_c48_l66_s0p5_m90": replace(
        DHJ_BASE,
        group="dhj_s",
        title=r"C48, x2 stretch @ -90$^\circ$",
        label="c48_s0p5_m90",
        stretch_factor=0.5,
        target_lon=-90,
        kw_plt={"color": "C3", "linestyle": "dashed", "dash_capstyle": "round"},
    ),
    "dhj_c48_l66_s0p5_p90": replace(
        DHJ_BASE,
        group="dhj_s",
        title=r"C48, x2 stretch @ +90$^\circ$",
        label="c48_s0p5_p90",
        stretch_factor=0.5,
        target_lon=90,
        kw_plt={"color": "C4", "linestyle": "dashed", "dash_capstyle": "round"},
    ),
    # hd209
    "hd209_dt30_nrad3": replace(HD209_BASE, label="dt30_nrad3", n_radstep=3),
    "hd209_dt30_nrad4": replace(HD209_BASE, label="dt30_nrad4"),
    "hd209_dt30_nrad5": replace(HD209_BASE, label="dt30_nrad5", n_radstep=5),
    # "hd209_int_flux_zero": replace(
    #     HD209_BASE, label="int_flux_zero", internal_flux=InternalFlux(value=0.0)
    # ),
    # "hd209_rand_t100k_nf20_tau1e5": replace(
    #     HD209_BASE,
    #     label="rand_t100k_nf20_tau1e5",
    #     internal_flux=InternalFlux(method="random_harmonics"),
    # ),
    # "hd209_rand_t300k_nf20_tau1e5": replace(
    #     HD209_BASE,
    #     label="rand_t300k_nf20_tau1e5",
    #     internal_flux=InternalFlux(method="random_harmonics", value=459.299727),
    # ),
    # "hd209_rand_t100k_nf20_tau1e5_amp5e-5": replace(
    #     HD209_BASE,
    #     label="rand_t100k_nf20_tau1e5_amp5e-5",
    #     internal_flux=InternalFlux(method="random_harmonics", heating_amplitude=5e-5),
    # ),
    # "hd209_rand_t100k_nf20_tau1e5_amp5e-5_sclht4": replace(
    #     HD209_BASE,
    #     label="rand_t100k_nf20_tau1e5_amp5e-5_sclht4",
    #     internal_flux=InternalFlux(
    #         method="random_harmonics", heating_amplitude=5e-5, scale_heights=4.0
    #     ),
    # ),
    # "hd209_rand_t100k_nf20_tau1e5_amp1e-3_sclht4": replace(
    #     HD209_BASE,
    #     label="rand_t100k_nf20_tau1e5_amp1e-3_sclht4",
    #     internal_flux=InternalFlux(
    #         method="random_harmonics", heating_amplitude=1e-3, scale_heights=4.0
    #     ),
    # ),
    "hd209_rand_t100k_nf20_tau1e5_amp1e-2_sclht6": replace(
        HD209_BASE,
        group="hd209_rand",
        label="rand_t100k_nf20_tau1e5_amp1e-2_sclht6",
        internal_flux=InternalFlux(
            method="random_harmonics", heating_amplitude=1e-2, scale_heights=6.0
        ),
    ),
    # "hd209_base_c48_nrad4": replace(
    #     HD209_BASE,
    #     label="base_c48_nrad4",
    #     run_length=1200,
    # ),
    # "hd209_base_c48_nrad4_new": replace(
    #     HD209_BASE,
    #     label="base_c48_nrad4_new",
    #     run_length=12000,
    # ),
}


AWS_EXPERIMENTS = {
    # SHJ, 24x16 processors
    "aws_shj_arm": replace(
        SHJ_BASE,
        title="SHJ C48 L32 (ARM)",
        group="aws-test",
        label="u-dr932-arm-1200-24x16",
    ),
    "aws_shj_x86": replace(
        SHJ_BASE,
        title="SHJ C48 L32 (x86)",
        group="aws-test",
        label="u-dr932-x86-1200-24x16",
    ),
    # DHJ
    "aws_dhj_arm": replace(
        DHJ_BASE,
        title="DHJ C48 (ARM)",
        group="aws-test",
        label="u-dr932-dhj-arm",
        run_length=100,
        timestep=50,
    ),
    "aws_dhj_x86": replace(
        DHJ_BASE,
        title="DHJ C48 (x86)",
        group="aws-test",
        label="u-dr932-dhj-x86",
        run_length=100,
        timestep=50,
    ),
    "aws_dhj_x86_fmoff": replace(
        DHJ_BASE,
        title="DHJ C48 (x86, fast-math off)",
        group="aws-test",
        label="u-dr932-dhj-x86-fmoff",
        run_length=100,
        timestep=50,
    ),
}


def exp_global_attrs(exp_key, experiments=EXPERIMENTS):
    """Global attributes of a netCDF file describing one experiment."""
    exp = experiments[exp_key]
    attrs = {
        **GLOBAL_ATTRS,
        "name": exp_key,
        "group": exp.group,
        "label": exp.label,
        "resolution": exp.resolution,
        "n_levels": exp.n_levels,
        "timestep_s": exp.timestep,
        "run_length_days": exp.run_length,
        "stretch_factor": exp.stretch_factor,
        "target_lon": exp.target_lon,
        "date_created": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
    }
    if exp.internal_flux is not None:
        for key, value in vars(exp.internal_flux).items():
            attrs[f"internal_flux_{key}"] = value
    return attrs


@dataclass
class BenchModel:
    """Model details."""

    title: str
    kw_plt: dict = field(default_factory=dict)
    model: Model = lfric
    details: str = ""
    datetime_start: datetime = datetime.strptime(TIME_ORIGIN, "%Y-%m-%d %H:%M:%S")  # noqa: RUF009


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


@dataclass
class Group:
    """Details for a group of simulations."""

    title: str
    simulations: Iterable
    kw_plt: dict = field(default_factory=dict)


GROUPS = {
    "shj": Group(
        title="Shallow Hot Jupiter",
        simulations=[k for k, v in EXPERIMENTS.items() if v.group == "shj"],
    ),
    "dhj": Group(
        title="Deep Hot Jupiter",
        simulations=[k for k, v in EXPERIMENTS.items() if v.group == "dhj"],
    ),
    "dhj_s": Group(
        title="Deep Hot Jupiter, Stretched Mesh",
        simulations=[k for k, v in EXPERIMENTS.items() if v.group == "dhj_s"],
    ),
    "dhj_all": Group(
        title="Deep Hot Jupiter",
        simulations=[k for k, v in EXPERIMENTS.items() if v.group in ["dhj", "dhj_s"]],
    ),
    "shj_dhj_all": Group(
        title="Shallow & Deep Hot Jupiter",
        simulations=[
            k for k, v in EXPERIMENTS.items() if v.group in ["shj", "dhj", "dhj_s"]
        ],
    ),
    "hd209_all": Group(
        title="HD 209458b (all)",
        simulations=[
            k for k, v in EXPERIMENTS.items() if v.group in ["hd209", "hd209_rand"]
        ],
    ),
    "hd209": Group(
        title="HD 209458b",
        simulations=[k for k, v in EXPERIMENTS.items() if v.group in ["hd209"]],
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


def setup_pressure_axis(ax, lim=None, *, linear=False):
    """Make the y axis a pressure axis increasing downwards, log by default.

    `lim` is the (min, max) pressure range, applied top-down.
    """
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("linear" if linear else "log")
    if lim is None:
        ax.invert_yaxis()
    else:
        ax.set_ylim(max(lim), min(lim))


def setup_latitude_axis(ax, which="y"):
    """Label and limit a latitude axis."""
    getattr(ax, f"set_{which}label")("Latitude [deg]")
    getattr(ax, f"set_{which}lim")(-90, 90)


def setup_longitude_axis(ax, which="x"):
    """Label and limit a longitude axis."""
    getattr(ax, f"set_{which}label")("Longitude [deg]")
    getattr(ax, f"set_{which}lim")(0, 360)


def interp_to_levels_of(cube, target, model=lfric):
    """Interpolate `cube` (e.g. air pressure) to the vertical levels of `target`.

    Used to get the pressure on the same grid as a diagnostic,
    just in case they are on different levels.
    """
    return cube.interpolate(
        [(model.z, target.coord(model.z).points)],
        iris.analysis.Linear(),  # type: ignore
    )


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


def file_end_date(fname):
    """End date of the output in a file named like `*_YYYYMMDD-YYYYMMDD.nc`."""
    return datetime.strptime(Path(fname).stem.split("_")[-1].split("-")[1], "%Y%m%d")


def load_exp_days(
    exp_key, start_day, end_day, file_glob=LATLON_GLOB, experiments=EXPERIMENTS
):
    """Load the output files of one experiment covering days `start_day`-`end_day`.

    Files are found in `data/lfric/<group>/<label>/`; a file is included if
    its end date is within (`start_day`, `end_day`].
    """
    exp = experiments[exp_key]
    fnames = sorted((paths.data_work / "lfric" / exp.group / exp.label).glob(file_glob))
    if not fnames:
        raise FileNotFoundError(f"No {file_glob} files found for {exp_key}")
    origin = datetime.strptime(TIME_ORIGIN, "%Y-%m-%d %H:%M:%S")
    end_days = [(file_end_date(f) - origin).days for f in fnames]
    if max(end_days) < end_day:
        raise ValueError(f"{exp_key} output ends on day {max(end_days)} < {end_day}")
    fnames = [f for f, d in zip(fnames, end_days) if start_day < d <= end_day]
    return load_lfric_raw(
        fnames,
        callback=partial(
            lfric_callback_uniform_height,
            model_top_height=exp.const.domain_height.data,
        ),
    )


def add_time_coord(cube, field, filename, time_origin=TIME_ORIGIN):
    """Extract time from filename and add as aux coord to cube."""
    dt = datetime.strptime(Path(filename).stem.split("_")[-1].split("-")[1], "%Y%m%d")
    dt_sec = (dt - datetime.strptime(time_origin, "%Y-%m-%d %H:%M:%S")).total_seconds()
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


def create_gaussian_cube(nlat=None, nlon=None, n_gauss=None, pm180=False, radius=None):
    """
    Create a dummy 2D cube on a regular Gaussian lat-lon grid.

    A counterpart of `aeolus.io.create_dummy_cube`, with latitudes at the
    Gauss-Legendre quadrature nodes instead of equally spaced. The latitude
    bounds are placed so that the area of each latitude band matches its
    quadrature weight, i.e. `sin(bound)` is the cumulative sum of the weights.
    Longitudes are equally spaced, starting at 0 (or -180 if `pm180`).

    Parameters
    ----------
    nlat: int, optional
        Number of points in latitude. Not needed if `n_gauss` is given.
    nlon: int, optional
        Number of points in longitude. Not needed if `n_gauss` is given.
    n_gauss: int, optional
        Gaussian grid number N, the number of latitudes between a pole and the
        equator. If given, `nlat = 2 * N` and `nlon = 4 * N`.
        Note that this is not the UM N-notation used by `create_dummy_cube`.
    pm180: bool, optional
        Use -/+180 instead of 0 to 360 for the longitude span.
    radius: float, optional
        Planet radius (m) of the coordinate system. Default: Earth radius.

    Returns
    -------
    iris.cube.Cube
        A dummy 2D cube of zeros.

    Examples
    --------
    >>> create_gaussian_cube(nlat=96, nlon=192)
    >>> create_gaussian_cube(n_gauss=48)
    """
    if n_gauss is not None:
        nlat = 2 * n_gauss
        nlon = 4 * n_gauss
    if nlat is None or nlon is None:
        msg = "Give either `n_gauss` or both `nlat` and `nlon`."
        raise ValueError(msg)

    geog_cs = GeogCS(EARTH_RADIUS if radius is None else radius)

    # Latitude: Gauss-Legendre nodes in sin(lat), ascending from south to north
    nodes, weights = np.polynomial.legendre.leggauss(nlat)
    lats = np.rad2deg(np.arcsin(nodes))
    sin_bounds = np.clip(np.concatenate([[-1.0], np.cumsum(weights) - 1.0]), -1, 1)
    sin_bounds[-1] = 1.0
    lat_bounds = np.rad2deg(np.arcsin(sin_bounds))
    lat_coord = DimCoord(
        lats,
        standard_name="latitude",
        units="degrees",
        bounds=np.column_stack([lat_bounds[:-1], lat_bounds[1:]]),
        coord_system=geog_cs,
    )

    # Longitude
    lons = np.arange(nlon) * 360.0 / nlon
    if pm180:
        lons -= 180.0
    lon_coord = DimCoord(
        lons,
        standard_name="longitude",
        units="degrees",
        circular=True,
        coord_system=geog_cs,
    )
    lon_coord.guess_bounds()

    return Cube(
        np.zeros((nlat, nlon), dtype=np.int32),
        dim_coords_and_dims=((lat_coord, 0), (lon_coord, 1)),
        var_name="dummy_cube",
        units="1",
    )
