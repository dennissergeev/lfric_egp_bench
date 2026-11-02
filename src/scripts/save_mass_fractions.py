"""
Extract the chemical species mass fractions from LFRic output of an HD 209458b run.

The lat-lon diagnostic files (`lfric_diag_latlon*.nc`) of a simulation in
`src/data/lfric/hd209/<label>/` are loaded and every `mass_fraction_of_*`
cube, together with `air_pressure` and `air_temperature`, is concatenated in
time and saved to a single netCDF file with global attributes describing the
experiment::

    src/data/lfric/hd209/<label>/<exp_key>_mass_fractions.nc

Only these variables are read from disk, so the other
diagnostics in the same files cost nothing.  The `full_levels` coordinate is
replaced with the equally spaced height levels, as in `quicklook.py`.

Examples
--------
Extract the mass fractions of one experiment::

    python save_mass_fractions.py -e hd209_base_c48

Process all HD 209458b experiments, compressing the output::

    python save_mass_fractions.py --all --zlib

Use only the last 20 files (i.e. the end of the run)::

    python save_mass_fractions.py -e hd209_base_c48 -n -20
"""

import warnings
from functools import partial
from pathlib import Path

import click
import iris
import paths
from aeolus.lfric import clean_attrs
from common import (
    EXPERIMENTS,
    LATLON_GLOB,
    chain_callbacks,
    exp_global_attrs,
    lfric_callback_uniform_height,
)

GROUP = "hd209"
OUT_SUFFIX = "mass_fractions"
NAME_PREFIX = "mass_fraction_of_"
# Thermodynamic fields saved alongside the mass fractions
EXTRA_NAMES = ("air_pressure", "air_temperature")
HD209_EXPERIMENTS = sorted(k for k, v in EXPERIMENTS.items() if v.group == GROUP)


def input_files(exp_key, file_glob=LATLON_GLOB, n_files=None):
    """Lat-lon diagnostic files of one experiment, in date order.

    A positive `n_files` keeps the first `n_files` files, a negative one the
    last `abs(n_files)` files.
    """
    exp = EXPERIMENTS[exp_key]
    exp_dir = paths.data_work / "lfric" / exp.group / exp.label
    files = sorted(exp_dir.glob(file_glob))
    if not files:
        msg = f"No files matching {file_glob} in {exp_dir}"
        raise click.ClickException(msg)
    if n_files is not None:
        files = files[:n_files] if n_files > 0 else files[n_files:]
    return files


def is_mass_fraction(cube):
    """True if `cube` is the mass fraction of a chemical species."""
    return cube.name().startswith(NAME_PREFIX)


def load_mass_fractions(exp_key, files):
    """Load and concatenate in time the mass fractions, pressure and temperature."""
    exp = EXPERIMENTS[exp_key]
    constraint = iris.Constraint(
        cube_func=lambda c: is_mass_fraction(c) or c.name() in EXTRA_NAMES
    )
    callback = chain_callbacks(
        partial(
            lfric_callback_uniform_height,
            model_top_height=exp.const.domain_height.data,  # type: ignore
        ),
        clean_attrs,
    )
    cubelist = iris.load(files, constraint, callback=callback)
    if not any(is_mass_fraction(cube) for cube in cubelist):
        msg = f"No {NAME_PREFIX}* cubes found in the files of {exp_key}"
        raise click.ClickException(msg)
    cubelist = cubelist.concatenate(check_aux_coords=False)
    names = [cube.name() for cube in cubelist]
    if missing := set(EXTRA_NAMES) - set(names):
        msg = f"{sorted(missing)} not found in the files of {exp_key}"
        raise click.ClickException(msg)
    if len(names) != len(set(names)):
        msg = f"Cubes of {exp_key} did not concatenate: {sorted(names)}"
        raise click.ClickException(msg)
    return iris.cube.CubeList(sorted(cubelist, key=lambda c: c.name()))  # type: ignore


def global_attrs(exp_key, cubelist, files):
    """Global attributes of the output file."""
    exp = EXPERIMENTS[exp_key]
    species = [
        cube.name().removeprefix(NAME_PREFIX).removesuffix("_in_air")
        for cube in cubelist
        if is_mass_fraction(cube)
    ]
    time = cubelist[0].coord("time")
    t_start, t_end = time.units.num2date(time.points[[0, -1]])
    attrs = exp_global_attrs(exp_key)
    attrs |= {
        "title": f"Mass fractions of chemical species in LFRic {exp.title} ({exp_key})",
        "summary": (
            f"Mass fractions of {len(species)} species ({', '.join(species)})"
            f" from the LFRic simulation {exp_key!r} of the hot Jupiter"
            " HD 209458b, with air pressure and temperature, extracted from"
            " the lat-lon diagnostic output."
        ),
        "planet": "HD 209458b",
        "species": " ".join(species),
        "time_coverage_start": f"{t_start}",
        "time_coverage_end": f"{t_end}",
        "history": (
            f"{attrs['date_created']}: created by {Path(__file__).name} from"
            f" {len(files)} files {files[0].name} ... {files[-1].name}"
        ),
    }
    return attrs


def save_experiment(exp_key, file_glob, n_files, outdir, overwrite, zlib):
    """Extract and save the mass fractions of one experiment."""
    exp = EXPERIMENTS[exp_key]
    outdir = outdir or (paths.data_work / "lfric" / exp.group / exp.label)
    fname_out = outdir / f"{exp_key}_{OUT_SUFFIX}.nc"
    if fname_out.is_file() and not overwrite:
        click.echo(f"Skipping (exists): {fname_out}")
        return
    files = input_files(exp_key, file_glob, n_files)
    cubelist = load_mass_fractions(exp_key, files)
    attrs = global_attrs(exp_key, cubelist, files)
    for cube in cubelist:
        # XIOS boilerplate that the new global attributes replace
        for key in ("description", "title", "name"):
            cube.attributes.globals.pop(key, None)
            cube.attributes.locals.pop(key, None)
        cube.attributes.globals.update(attrs)
    fname_out.parent.mkdir(parents=True, exist_ok=True)
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(cubelist, fname_out, zlib=zlib)
    click.echo(f"Saved: {fname_out}")


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.option(
    "-e",
    "--exp",
    "exp_keys",
    type=click.Choice(HD209_EXPERIMENTS),
    multiple=True,
    help="Experiment key; repeatable.",
)
@click.option(
    "--all", "all_exps", is_flag=True, help=f"Run for all experiments in {GROUP!r}."
)
@click.option(
    "--glob",
    "file_glob",
    default=LATLON_GLOB,
    help="Glob for the input files within the experiment directory.",
)
@click.option(
    "-n",
    "--n-files",
    type=int,
    default=None,
    help="Load only the first N files, or the last N if negative. Default: all.",
)
@click.option(
    "--zlib/--no-zlib",
    default=False,
    help="Compress the output variables.",
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
def main(exp_keys, all_exps, **kwargs):
    """
    Save the mass_fraction_of_* cubes of HD 209458b runs to a netCDF file.

    \b
    Examples:
      save_mass_fractions.py -e hd209_base_c48
      save_mass_fractions.py -e hd209_base_c48 -e hd209_int_flux_zero --zlib
      save_mass_fractions.py --all -o /scratch/mass_fractions
      save_mass_fractions.py -e hd209_base_c48 -n -20
    """
    all_exp_keys = dict.fromkeys(HD209_EXPERIMENTS if all_exps else exp_keys)
    if not all_exp_keys:
        msg = "Specify at least one experiment (-e) or --all."
        raise click.UsageError(msg)
    if kwargs["n_files"] == 0:
        msg = "--n-files must be non-zero."
        raise click.UsageError(msg)
    for exp_key in all_exp_keys:
        click.echo(f"Processing {exp_key}")
        save_experiment(exp_key, **kwargs)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iris.FUTURE.date_microseconds = True
        main()
