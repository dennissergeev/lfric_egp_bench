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

The vertical `full_levels`/`half_levels` coordinates are left alone unless
`--interp-vertically` is given, which puts every cube on the `full_levels`
height grid of the reference cube.

Examples
--------
Regrid one experiment to the default 144x90 grid::

    python regrid.py -e shj_c48_l32

Regrid a whole group to a finer grid, overwriting existing output::

    python regrid.py -g dhj --nlon 288 --nlat 180 --overwrite
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
from common import CATEGORIES, EXPERIMENTS, GROUPS, PROJECT
from iris.mesh import MeshXY

# `iris` refuses to build a `GeogCS` for a non-Earth radius without this
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

RAW_GLOB = "lfric_diag_main*.nc"
RAW_STEM = "lfric_diag_main"
OUT_STEM = "lfric_diag_regr"
# Cube used to define the mesh and, with --interp-vertically, the target levels
REF_CUBE_NAME = "air_temperature"


@dataclass
class RegridOpts:
    """Options controlling the target grid and the output files."""

    nlat: int
    nlon: int
    n_res: int | None
    pm180: bool
    ref_cube_name: str
    interp_vertically: bool
    overwrite: bool
    outdir: Path | None


def make_target_cube(opts):
    """Dummy cube defining the target lat-lon grid."""
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

    The target grid dimensions are embedded in the output stem so that
    files regridded to different resolutions do not overwrite each other.

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
    out_stem = f"{OUT_STEM}_nx{nlon}_ny{nlat}"
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


def regrid_cubelist(cubelist, tgt_cube, opts, regridder=None):
    """Regrid every mesh cube of `cubelist`, reusing `regridder` if given.

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
    for cube in cubelist:
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
        "name": exp_key,
        "project": PROJECT,
        "group": exp.group,
        "label": exp.label,
        "processed": "True",
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
    tgt_cube = make_target_cube(opts)
    for exp_key in all_exp_keys:
        click.echo(f"Regridding {exp_key}")
        regrid_experiment(exp_key, tgt_cube, opts)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iris.FUTURE.date_microseconds = True
        main()
