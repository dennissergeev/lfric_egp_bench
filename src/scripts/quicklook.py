"""
Quick-look overview of one LFRic diagnostic on the lat-lon output.

The regridded diagnostic files (`lfric_diag_latlon*.nc`) of a simulation in
`src/data/lfric/<group>/<label>/` are loaded, interpolated to a target
pressure level and shown as a four-panel summary:

- `z-y`: time and zonal mean cross-section;
- `t-z`: zonal mean at the equator, in time;
- `y-x`: time mean map at the target pressure, with streamlines;
- `t-y`: zonal mean at the target pressure, in time.

Examples
--------
Plot the eastward wind of one experiment::

    python quicklook.py -e hd209_rand_t100k_nf20_tau1e5

Plot the temperature at 0.1 bar, averaged over the last 100 days, with
explicit colour limits::

    python quicklook.py -e shj_c48_l32 -d temp -p 10000 -n 100 \\
        --norm linear --vmin 500 --vmax 1600 --cmap magma
"""  # noqa: EXE002

import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import dask.array as da
import iris
import matplotlib.colors as mcol
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import paths
from aeolus.calc import last_n_day_mean, superrotation_index, zonal_mean
from aeolus.coord import (
    get_cube_rel_days,
    interp_cube_from_height_to_pressure_levels,
)
from aeolus.lfric import load_lfric_raw
from aeolus.model import lfric
from aeolus.plot import capitalise, figsave, stream, subplot_label_generator
from common import (
    CMAPS,
    EXPERIMENTS,
    GROUPS,
    KW_REF_LINE,
    LAT_TICKS,
    LATLON_GLOB,
    LON_TICKS,
    PA_TO_BAR,
    interp_to_levels_of,
    lfric_callback_uniform_height,
    setup_latitude_axis,
    setup_longitude_axis,
    setup_pressure_axis,
)

# Colour map of diagnostics without an entry in `common.CMAPS`
DEFAULT_CMAP = "RdBu_r"
# Panels that can show contours of the superrotation index
SUPERROT_PANELS = ("z-y", "t-y")
# Style of the superrotation index contours
KW_SUPERROT = {
    "colors": "C2",
    "linewidths": 1,
    "path_effects": [mpe.withStroke(linewidth=2.5, foreground="w")],
}


@dataclass
class PlotOpts:
    """Options controlling what is plotted and how."""

    diag_key: str
    target_pressure: float
    n_days_mean: float
    sim_days: float | None
    method_plt: str
    n_contours: int
    cmap: str | None
    norm_kind: str
    vmin: float | None
    vmax: float | None
    vcenter: float
    do_stream: bool
    linear_pressure: bool
    superrot_panels: tuple[str, ...]
    superrot_levels: tuple[float, ...]
    outdir: Path | None


def load_dataset(exp_key, file_glob=LATLON_GLOB):
    """Load the lat-lon diagnostic files of one experiment."""
    exp = EXPERIMENTS[exp_key]
    exp_dir = paths.data_work / "lfric" / exp.group / exp.label
    files = sorted(exp_dir.glob(file_glob))
    if not files:
        msg = f"No files matching {file_glob} in {exp_dir}"
        raise click.ClickException(msg)
    return load_lfric_raw(
        files,
        callback=partial(
            lfric_callback_uniform_height,
            model_top_height=exp.const.domain_height.data,  # type: ignore
        ),
    )


def extract_cube(dset, diag_key):
    """Extract a cube by its short name in the aeolus model container."""
    try:
        name = getattr(lfric, diag_key)
    except AttributeError as e:
        msg = f"{diag_key!r} is not a field of the lfric model container"
        raise click.ClickException(msg) from e
    try:
        return dset.extract_cube(name)
    except iris.exceptions.ConstraintMismatchError as e:  # type: ignore
        msg = f"{name!r} ({diag_key}) not found in the data"
        raise click.ClickException(msg) from e


def centre_on_lon(cube, lon0, model=lfric):
    """
    Shift the longitude of `cube` to be relative to `lon0`, within -180...180.

    The data are rolled along the longitude dimension accordingly,
    so that `lon0` ends up at 0 deg in the middle of the domain.
    """
    xcoord = cube.coord(model.x)
    (xdim,) = cube.coord_dims(xcoord)
    rel = (xcoord.points - lon0 + 180) % 360 - 180
    # Bring the westernmost relative longitude to the start
    shift = -int(np.argmin(rel))
    offset = np.roll(xcoord.points - rel, shift)
    bounds = None
    if xcoord.has_bounds():
        bounds = np.roll(xcoord.bounds, shift, axis=0) - offset[:, None]
    new = cube.copy(data=da.roll(cube.core_data(), shift, axis=xdim))
    new.replace_coord(xcoord.copy(points=np.roll(rel, shift), bounds=bounds))
    return new


def calc_superrotation_index(u, exp):
    """Local superrotation index (Read, 1986) from the zonal wind cube."""
    return superrotation_index(iris.cube.CubeList([u]), const=exp.const, model=lfric)


def add_superrot_contours(ax, x, y, s_idx, opts):
    """Overlay contours of the superrotation index on `ax`."""
    cs = ax.contour(x, y, s_idx, levels=sorted(opts.superrot_levels), **KW_SUPERROT)
    labels = ax.clabel(cs, fmt=lambda x: f"SI={x:g}", fontsize="xx-small")
    for label in labels:
        label.set_path_effects(KW_SUPERROT["path_effects"])


def make_norm(opts):
    """Colour normalisation, either centred on `vcenter` or plain linear."""
    if opts.norm_kind == "linear":
        return mcol.Normalize(vmin=opts.vmin, vmax=opts.vmax)
    if opts.vmin is None and opts.vmax is None:
        return mcol.CenteredNorm(vcenter=opts.vcenter)
    return mcol.TwoSlopeNorm(vcenter=opts.vcenter, vmin=opts.vmin, vmax=opts.vmax)


def plot_quicklook(dset, exp_key, opts):
    """Assemble the four-panel quick-look figure for one experiment."""
    exp = EXPERIMENTS[exp_key]
    sim_days = exp.run_length if opts.sim_days is None else opts.sim_days

    var = extract_cube(dset, opts.diag_key)
    u = extract_cube(dset, "u")
    v = extract_cube(dset, "v")

    # Interpolate pressure to the vertical grid of `var`
    # just in case they are on different levels
    pres = extract_cube(dset, "pres")
    pres_interp_var = interp_to_levels_of(pres, var)
    # Centre the pressure-level fields on the substellar point,
    # with longitudes relative to it within -180...180
    def to_plev(cube, pres_interp):
        cube_plev = interp_cube_from_height_to_pressure_levels(
            cube, pres_interp, [opts.target_pressure]
        )
        return centre_on_lon(cube_plev, exp.substellar_lon)

    var_plev = to_plev(var, pres_interp_var)

    if opts.do_stream or "t-y" in opts.superrot_panels:
        pres_interp_uv = interp_to_levels_of(pres, u)
        u_plev = to_plev(u, pres_interp_uv)
    if opts.do_stream:
        v_plev = to_plev(v, pres_interp_uv)
    # Vertical coordinate of the pressure-axis panels: the time and zonal
    # mean pressure on the (z, y) grid, used as such for the cross-section
    # and sliced at the equator for the time-height panel.
    pres_zm_tm = last_n_day_mean(zonal_mean(pres_interp_var), opts.n_days_mean)  # type: ignore

    lats = var.coord(lfric.y).points
    lons = var_plev.coord(lfric.x).points
    days = get_cube_rel_days(var)
    pres_bar_zy = pres_zm_tm.data * PA_TO_BAR  # type: ignore
    lats_zy = np.broadcast_to(lats, pres_bar_zy.shape)
    # Latitude closest to the equator, shared by the time-height panel and
    # its vertical axis so that both pressure axes match exactly
    idx_eq = np.abs(lats).argmin()
    lat_eq = lats[idx_eq]
    pres_bar_eq = pres_bar_zy[:, idx_eq]
    # Common vertical range of the two pressure-axis panels
    pres_lim = (pres_bar_zy.min(), pres_bar_zy.max())
    target_bar = opts.target_pressure * PA_TO_BAR
    var_zm_tm = last_n_day_mean(zonal_mean(var), opts.n_days_mean).data  # type: ignore
    cmap = opts.cmap or CMAPS.get(opts.diag_key, DEFAULT_CMAP)
    kw_plt = {"norm": make_norm(opts), "cmap": cmap, "rasterized": True}

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    axd = fig.subplot_mosaic(
        [["z-y", "t-z", "cbar"], ["y-x", "t-y", "cbar"]],
        gridspec_kw={"width_ratios": [1, 1, 0.02]},
    )

    iletters = subplot_label_generator()

    ax = axd["z-y"]
    ax.set_title(f"({next(iletters)})", loc="left")
    ax.set_title(f"{opts.n_days_mean:.0f}-day and zonal mean")
    setup_latitude_axis(ax, which="x")
    ax.set_xticks(LAT_TICKS)
    setup_pressure_axis(ax, pres_lim, linear=opts.linear_pressure)
    im = getattr(ax, opts.method_plt)(lats_zy, pres_bar_zy, var_zm_tm, **kw_plt)
    if opts.n_contours:
        cs = ax.contour(
            lats_zy,
            pres_bar_zy,
            var_zm_tm,
            opts.n_contours,
            colors="k",
            linewidths=0.5,
        )
        ax.clabel(cs, fmt="%d", fontsize="xx-small")
    if "z-y" in opts.superrot_panels:
        # Zonal wind on the levels of `var`, to share the vertical coordinate
        u_zm_tm = last_n_day_mean(zonal_mean(interp_to_levels_of(u, var)), opts.n_days_mean)  # type: ignore
        add_superrot_contours(
            ax, lats_zy, pres_bar_zy, calc_superrotation_index(u_zm_tm, exp).data, opts
        )
    ax.axhline(target_bar, **KW_REF_LINE)

    ax = axd["t-z"]
    ax.set_title(f"({next(iletters)})", loc="left")
    ax.set_title("Zonal mean, equator")
    setup_pressure_axis(ax, pres_lim, linear=opts.linear_pressure)
    ax.set_xlabel("Time [days]")
    ax.set_xlim(0, sim_days)
    im = getattr(ax, opts.method_plt)(
        days,
        pres_bar_eq,
        iris.util.squeeze(  # type: ignore
            zonal_mean(var.interpolate([(lfric.y, [lat_eq])], iris.analysis.Linear()))  # type: ignore
        ).data.T,
        **kw_plt,
    )
    ax.axhline(target_bar, **KW_REF_LINE)
    ax.axvline(sim_days - opts.n_days_mean, **KW_REF_LINE)

    ax = axd["y-x"]
    ax.set_title(f"({next(iletters)})", loc="left")
    ax.set_title(
        f"{opts.n_days_mean:.0f}-day mean, {target_bar:>5.2f} bar",
    )
    setup_latitude_axis(ax)
    ax.set_yticks(LAT_TICKS)
    setup_longitude_axis(ax, pm180=True)
    ax.set_xticks(LON_TICKS)
    if exp.substellar_lon % 360:
        ax.set_xlabel("Longitude from substellar point [deg]")
    im = getattr(ax, opts.method_plt)(
        lons,
        lats,
        last_n_day_mean(var_plev, opts.n_days_mean).data,  # type: ignore
        **kw_plt,
    )
    if opts.do_stream:
        stream(
            last_n_day_mean(u_plev, opts.n_days_mean),  # type: ignore
            last_n_day_mean(v_plev, opts.n_days_mean),  # type: ignore
            color="k",
            density=0.75,
            arrowsize=0.5,
            linewidth=0.5,
            ax=ax,
        )

    ax = axd["t-y"]
    ax.set_title(f"({next(iletters)})", loc="left")
    ax.set_title(f"Zonal mean, {target_bar:>5.2f} bar")
    setup_latitude_axis(ax)
    ax.set_yticks(LAT_TICKS)
    ax.set_xlabel("Time [days]")
    ax.set_xlim(0, sim_days)
    im = getattr(ax, opts.method_plt)(days, lats, zonal_mean(var_plev).data.T, **kw_plt)
    if "t-y" in opts.superrot_panels:
        s_idx = calc_superrotation_index(zonal_mean(u_plev), exp)  # type: ignore
        add_superrot_contours(ax, days, lats, iris.util.squeeze(s_idx).data.T, opts)  # type: ignore
    ax.axvline(sim_days - opts.n_days_mean, **KW_REF_LINE)

    fig.colorbar(im, cax=axd["cbar"], orientation="vertical")
    fig.suptitle(
        "\n".join([
            f"{exp.title} | LFRic-Atmosphere",
            f"{capitalise(var.name())} [{var.units}]",
        ])
    )
    return fig


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
    "-d",
    "--diag",
    "diag_key",
    default="u",
    help="Diagnostic to plot, as named in the aeolus lfric model container.",
)
@click.option(
    "-p",
    "--pressure",
    "target_pressure",
    type=float,
    default=10_000.0,
    help="Target pressure level [Pa].",
)
@click.option(
    "-n",
    "--n-days-mean",
    type=float,
    default=200.0,
    help="Length of the averaging period at the end of the run [days].",
)
@click.option(
    "--sim-days",
    type=float,
    default=None,
    help="Length of the time axis [days]. Taken from the experiment if unset.",
)
@click.option(
    "--method",
    "method_plt",
    type=click.Choice(["pcolormesh", "contourf"]),
    default="pcolormesh",
    help="Plotting method for the shaded fields.",
)
@click.option(
    "--n-contours",
    type=click.IntRange(min=0),
    default=7,
    help="Number of contours over the z-y panel; 0 to disable.",
)
@click.option(
    "--cmap",
    default=None,
    help=f"Colour map. Default: from `common.CMAPS` by diagnostic, else {DEFAULT_CMAP}.",
)
@click.option(
    "--norm",
    "norm_kind",
    type=click.Choice(["centered", "linear"]),
    default="centered",
    help="Colour normalisation: centred on --vcenter, or plain linear.",
)
@click.option("--vmin", type=float, default=None, help="Lower colour limit.")
@click.option("--vmax", type=float, default=None, help="Upper colour limit.")
@click.option(
    "--vcenter",
    type=float,
    default=0.0,
    help="Centre of a centred colour normalisation.",
)
@click.option(
    "--stream/--no-stream",
    "do_stream",
    default=True,
    help="Overlay streamlines on the y-x panel.",
)
@click.option(
    "--linear-pressure/--log-pressure",
    "linear_pressure",
    default=False,
    help="Use a linear (instead of logarithmic) pressure axis.",
)
@click.option(
    "--superrot",
    "superrot_panels",
    type=click.Choice(SUPERROT_PANELS),
    multiple=True,
    help="Overlay contours of the superrotation index on this panel; repeatable.",
)
@click.option(
    "--superrot-level",
    "superrot_levels",
    type=float,
    multiple=True,
    default=(0.0,),
    help="Contour level of the superrotation index; repeatable.",
)
@click.option(
    "--glob",
    "file_glob",
    default=LATLON_GLOB,
    help="Glob for the input files within the experiment directory.",
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Default: src/figures/drafts/EXP/.",
)
def main(exp_keys, group_keys, file_glob, **kwargs):
    """
    Plot a four-panel quick-look overview of one LFRic diagnostic.

    Experiments may be selected individually with -e, or in bulk by group
    (-g); the two options combine and duplicates are dropped.

    \b
    Examples:
      quicklook.py -e hd209_rand_t100k_nf20_tau1e5
      quicklook.py -e shj_c48_l32 -d temp -p 10000 --norm linear --cmap magma
      quicklook.py -g shj -g dhj -d temp
      quicklook.py -e shj_c48_l32 --superrot z-y --superrot t-y
      quicklook.py -e shj_c48_l32 --glob "lfric_diag_regr*.nc"
    """
    all_exp_keys = dict.fromkeys(exp_keys)
    for group_key in dict.fromkeys(group_keys):
        all_exp_keys.update(dict.fromkeys(GROUPS[group_key].simulations))
    if not all_exp_keys:
        msg = "Specify at least one experiment (-e) or group (-g)."
        raise click.UsageError(msg)

    opts = PlotOpts(**kwargs)
    for exp_key in all_exp_keys:
        dset = load_dataset(exp_key, file_glob)
        fig = plot_quicklook(dset, exp_key, opts)
        outdir = opts.outdir or paths.drafts / "quicklooks"
        figsave(fig, outdir / f"{exp_key}_{opts.diag_key}_quicklook")
        plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iris.FUTURE.date_microseconds = True
        main()
