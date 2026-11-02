"""
Regrid raw LFRic output to a regular lat-lon grid and save it as netCDF.

The raw diagnostic files (`lfric_diag_main*.nc`) of a simulation in
`src/data/lfric/<group>/<label>/` are written by XIOS on the native
cubed-sphere mesh: a flat `cell` dimension with latitude and longitude
auxiliary coordinates carrying four-vertex bounds.  An `iris` mesh is
rebuilt from those bounds and each cube is regridded conservatively onto a
target lat-lon grid with `MeshToGridESMFRegridder`.

One output file is written per input file, mirroring its date span, so the
result concatenates the same way the raw data does::

    lfric_diag_main_20000101-20000111.nc -> lfric_diag_regr_nx144_ny90_20000101-20000111.nc

With `--gaussian` the target grid has Gaussian instead of equally spaced
latitudes (see `common.create_gaussian_cube`), and `_gauss` is added to the
output stem, e.g. `lfric_diag_regr_gauss_nx192_ny96_20000101-20000111.nc`.

By default it includes only: vorticity, divergence and the horizontal wind components;
select others with `--var` or keep everything with `--all-vars`.
The reference cube (`--ref-cube`) is always read, since it defines the source mesh,
but is saved only if selected.

The vertical `full_levels`/`half_levels` coordinates are left alone unless
`--interp-vertically` is given, which puts every cube on the `full_levels`
height grid of the reference cube.

Examples
--------
Regrid one experiment to the default 144x90 grid::

    python regrid.py -e shj_c48_l32

Regrid a whole group to a finer grid, overwriting existing output::

    python regrid.py -g dhj --nlon 288 --nlat 180 --overwrite

Regrid one experiment to a 192x96 Gaussian grid::

    python regrid.py -e dhj_c48_l66 --gaussian --nlon 192 --nlat 96

or, equivalently, with the Gaussian grid number N (`nlat = 2N`, `nlon = 4N`)::

    python regrid.py -e dhj_c48_l66 --n-gauss 48
"""  # noqa: EXE002

import os
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import iris
import paths
from aeolus.const import add_planet_conf_to_cubes
from aeolus.io import create_dummy_cube, save_cubelist
from aeolus.lfric import (
    add_equally_spaced_height_coord,
    load_lfric_raw,
    replace_level_coord_with_height,
)
from aeolus.model import lfric
from common import (
    CATEGORIES,
    EXPERIMENTS,
    GROUPS,
    create_gaussian_cube,
    exp_global_attrs,
)
from iris.mesh import MeshXY

# `iris` refuses to build a `GeogCS` for a non-Earth radius without this
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

RAW_GLOB = "lfric_diag_main*.nc"
RAW_STEM = "lfric_diag_main"
OUT_STEM = "lfric_diag_regr"
# Cube used to define the mesh and, with --interp-vertically, the target levels
REF_CUBE_NAME = "pressure_in_wth"
# Cubes saved by default
DEF_VAR_NAMES = (
    "vertical_vorticity",
    "divergence_of_wind",
    "u_in_w3",
    "v_in_w3",
    "pressure_in_wth",
)


@dataclass
class RegridOpts:
    """Options controlling the target grid and the output files."""

    nlat: int
    nlon: int
    n_res: int | None
    n_gauss: int | None
    gaussian: bool
    pm180: bool
    ref_cube_name: str
    var_names: tuple[str, ...]
    all_vars: bool
    interp_vertically: bool
    overwrite: bool
    outdir: Path | None


def make_target_cube(opts):
    """Dummy cube defining the target lat-lon grid."""
    if opts.gaussian:
        if opts.n_gauss is not None:
            return create_gaussian_cube(n_gauss=opts.n_gauss, pm180=opts.pm180)
        return create_gaussian_cube(nlat=opts.nlat, nlon=opts.nlon, pm180=opts.pm180)
    if opts.n_res is not None:
        return create_dummy_cube(n_res=opts.n_res, pm180=opts.pm180)
    return create_dummy_cube(nlat=opts.nlat, nlon=opts.nlon, pm180=opts.pm180)


def input_files(exp_key):
    """Raw diagnostic files of one experiment, in date order."""
    exp = EXPERIMENTS[exp_key]
    exp_dir = paths.data_work / "lfric" / exp.group / exp.label
    files = sorted(exp_dir.glob(RAW_GLOB))
    if not files:
        msg = f"No files matching {RAW_GLOB} in {exp_dir}"
        raise click.ClickException(msg)
    return files


def output_file(fname, exp_key, opts, tgt_cube):
    """Output path for one input file, keeping its date span.

    The target grid type and dimensions are embedded in the output stem so
    that files regridded to different grids do not overwrite each other.

    With an explicit `--outdir` the `<group>/<label>/` layout of the input
    tree is reproduced under it, so that experiments sharing a date span do
    not overwrite each other.
    """
    if opts.outdir is None:
        outdir = fname.parent
    else:
        exp = EXPERIMENTS[exp_key]
        outdir = opts.outdir / exp.group / exp.label
    nlat = tgt_cube.coord("latitude").shape[0]
    nlon = tgt_cube.coord("longitude").shape[0]
    grid = "_gauss" if opts.gaussian else ""
    out_stem = f"{OUT_STEM}{grid}_nx{nlon}_ny{nlat}"
    return outdir / fname.name.replace(RAW_STEM, out_stem, 1)


def horizontal_dim(cube):
    """Dimension of `cube` spanned by the unstructured cell axis, if any."""
    lat = cube.coords("latitude")
    lon = cube.coords("longitude")
    if not (lat and lon):
        return None
    dims = cube.coord_dims(lat[0])
    if len(dims) != 1 or cube.coord_dims(lon[0]) != dims:
        return None
    return dims[0]


def build_mesh(cube):
    """Rebuild the cubed-sphere mesh from the cell-corner bounds of `cube`."""
    lon = cube.coord("longitude")
    lat = cube.coord("latitude")
    if lon.bounds is None or lat.bounds is None:
        msg = (
            f"{cube.name()!r} has no cell bounds; conservative regridding needs"
            " the four-vertex bounds written by XIOS."
        )
        raise click.ClickException(msg)
    return MeshXY.from_coords(lon, lat)


def attach_mesh(cube, mesh, dim):
    """Swap the lat-lon aux coords of `cube` for the face coords of `mesh`."""
    out = cube.copy()
    out.remove_coord("latitude")
    out.remove_coord("longitude")
    for mesh_coord in mesh.to_MeshCoords(location="face"):
        out.add_aux_coord(mesh_coord, dim)
    return out


def cube_names(cube):
    """All names a cube can be selected by: standard, long and variable name."""
    return {cube.standard_name, cube.long_name, cube.var_name} - {None}


def select_cubes(cubelist, opts):
    """Cubes of `cubelist` to be saved, as chosen by `--var`/`--all-vars`."""
    if opts.all_vars:
        return cubelist
    selected = iris.cube.CubeList(  # type: ignore
        cube for cube in cubelist if cube_names(cube) & set(opts.var_names)
    )
    found = set().union(*(cube_names(cube) for cube in selected))
    if missing := set(opts.var_names) - found:
        msg = f"Variables not found in the data: {sorted(missing)}"
        raise click.ClickException(msg)
    return selected


def regrid_cubelist(cubelist, tgt_cube, opts, regridder=None):
    """Regrid the selected mesh cubes of `cubelist`, reusing `regridder` if given.

    Returns the regridded cube list and the regridder, so that the weights
    are computed once and reused across the files of an experiment.
    """
    from esmf_regrid.experimental.unstructured_scheme import MeshToGridESMFRegridder

    try:
        ref_cube = cubelist.extract_cube(opts.ref_cube_name)
    except iris.exceptions.ConstraintMismatchError as e:  # type: ignore
        msg = f"Reference cube {opts.ref_cube_name!r} not found in the data"
        raise click.ClickException(msg) from e
    ref_dim = horizontal_dim(ref_cube)
    if ref_dim is None:
        msg = f"Reference cube {opts.ref_cube_name!r} is not on the cell mesh"
        raise click.ClickException(msg)
    mesh = build_mesh(ref_cube)
    if regridder is None:
        regridder = MeshToGridESMFRegridder(
            attach_mesh(ref_cube, mesh, ref_dim),
            tgt_cube,
            method="conservative",  # type: ignore
        )

    result = iris.cube.CubeList()  # type: ignore
    for cube in select_cubes(cubelist, opts):
        dim = horizontal_dim(cube)
        if dim is None:
            # Nothing to regrid, e.g. a scalar or purely vertical diagnostic
            result.append(cube)
            continue
        result.append(regridder(attach_mesh(cube, mesh, dim)))

    if opts.interp_vertically:
        ref_z = replace_level_coord_with_height(ref_cube).coord(lfric.z)
        tgt_points = (lfric.z, ref_z.points)
        interp = iris.cube.CubeList()  # type: ignore
        for cube in result:
            if any(c.name().endswith("_levels") for c in cube.dim_coords):
                interp.append(
                    replace_level_coord_with_height(cube).interpolate(
                        [tgt_points],
                        iris.analysis.Linear(),  # type: ignore
                    )
                )
            else:
                interp.append(cube)
        result = interp
    return result, regridder


def regrid_experiment(exp_key, tgt_cube, opts):
    """Regrid and save every raw diagnostic file of one experiment."""
    exp = EXPERIMENTS[exp_key]
    gl_attrs = {
        **exp_global_attrs(exp_key),
        "processed": "True",
        "grid": "gaussian" if opts.gaussian else "regular",
    }
    # Vertical interpolation needs a `level_height` coord to interpolate onto.
    # It is added alongside the `full_levels`/`half_levels` coords rather than
    # replacing them, because those still identify which cubes need it.
    callback = None
    if opts.interp_vertically:
        callback = partial(
            add_equally_spaced_height_coord,
            model_top_height=exp.const.domain_height.data,  # type: ignore
        )
    regridder = None
    for fname in input_files(exp_key):
        fname_out = output_file(fname, exp_key, opts, tgt_cube)
        if fname_out.is_file() and not opts.overwrite:
            click.echo(f"Skipping (exists): {fname_out}")
            continue
        cubelist = load_lfric_raw([fname], callback=callback)
        dset_regr, regridder = regrid_cubelist(cubelist, tgt_cube, opts, regridder)
        add_planet_conf_to_cubes(dset_regr, const=exp.const)  # type: ignore
        fname_out.parent.mkdir(parents=True, exist_ok=True)
        save_cubelist(dset_regr, fname_out, **gl_attrs)
        click.echo(f"Saved: {fname_out}")


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.option(
    "-e",
    "--exp",
    "exp_keys",
    type=click.Choice(sorted(EXPERIMENTS)),
    multiple=True,
    help="Experiment key; repeatable.",
)
@click.option(
    "-g",
    "--group",
    "group_keys",
    type=click.Choice(sorted(GROUPS)),
    multiple=True,
    help="Run for all experiments in this group; repeatable.",
)
@click.option(
    "-c",
    "--category",
    "category_keys",
    type=click.Choice(sorted(CATEGORIES)),
    multiple=True,
    help="Run for all experiments in this category; repeatable.",
)
@click.option("--nlat", type=int, default=90, help="Target grid points in latitude.")
@click.option("--nlon", type=int, default=144, help="Target grid points in longitude.")
@click.option(
    "--n-res",
    type=int,
    default=None,
    help="Target grid in UM N-notation; overrides --nlat and --nlon.",
)
@click.option(
    "--n-gauss",
    type=int,
    default=None,
    help=(
        "Gaussian grid number N (nlat=2N, nlon=4N); implies --gaussian and"
        " overrides --nlat and --nlon."
    ),
)
@click.option(
    "--gaussian/--regular",
    default=False,
    help="Use Gaussian latitudes; the size is set by --nlat and --nlon.",
)
@click.option(
    "--pm180/--no-pm180",
    default=False,
    help="Span longitudes -180 to 180 instead of 0 to 360.",
)
@click.option(
    "--ref-cube",
    "ref_cube_name",
    default=REF_CUBE_NAME,
    help="Cube defining the source mesh and the target vertical levels.",
)
@click.option(
    "-v",
    "--var",
    "var_names",
    multiple=True,
    default=DEF_VAR_NAMES,
    help="Standard, long or variable name of a cube to save; repeatable.",
)
@click.option(
    "--all-vars/--selected-vars",
    default=False,
    help="Save all cubes, ignoring --var.",
)
@click.option(
    "--interp-vertically/--no-interp-vertically",
    default=False,
    help="Put all cubes on the height levels of the reference cube.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rewrite output files that already exist.",
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Default: alongside the input files.",
)
def main(exp_keys, group_keys, category_keys, **kwargs):
    """
    Regrid raw LFRic output to a regular lat-lon grid and save it as netCDF.

    Experiments may be selected individually with -e, or in bulk by group
    (-g) or category (-c); the three options combine and duplicates are
    dropped.

    \b
    Examples:
      regrid.py -e shj_c48_l32
      regrid.py -e dhj_c48_l66 --n-res 96 --overwrite
      regrid.py -e dhj_c48_l66 --gaussian --nlat 96 --nlon 192
      regrid.py -e dhj_c48_l66 --n-gauss 48
      regrid.py -e dhj_c48_l66 -v air_temperature -v air_pressure
      regrid.py -e dhj_c48_l66 --all-vars
      regrid.py -g shj -g dhj
      regrid.py -c tf -o /scratch/regridded
    """
    all_exp_keys = dict.fromkeys(exp_keys)
    for group_key in dict.fromkeys(group_keys):
        all_exp_keys.update(dict.fromkeys(GROUPS[group_key].simulations))
    for category_key in dict.fromkeys(category_keys):
        all_exp_keys.update(dict.fromkeys(CATEGORIES[category_key].simulations))
    if not all_exp_keys:
        msg = "Specify at least one experiment (-e), group (-g), or category (-c)."
        raise click.UsageError(msg)

    opts = RegridOpts(**kwargs)
    if opts.n_gauss is not None:
        opts.gaussian = True
    if opts.gaussian and opts.n_res is not None:
        msg = (
            "--n-res is the UM N-notation; use --n-gauss, or --nlat and --nlon,"
            " with --gaussian."
        )
        raise click.UsageError(msg)
    tgt_cube = make_target_cube(opts)
    for exp_key in all_exp_keys:
        click.echo(f"Regridding {exp_key}")
        regrid_experiment(exp_key, tgt_cube, opts)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iris.FUTURE.date_microseconds = True
        main()
