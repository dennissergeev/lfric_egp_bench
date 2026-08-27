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
import iris
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np
import paths
from aeolus.calc import last_n_day_mean, zonal_mean
from aeolus.coord import get_cube_rel_days, interp_cube_from_height_to_pressure_levels
from aeolus.lfric import load_lfric_raw
from aeolus.model import lfric
from aeolus.plot import figsave, stream
from common import CATEGORIES, EXPERIMENTS, GROUPS, lfric_callback_uniform_height

LATLON_GLOB = "lfric_diag_latlon*.nc"
PA_TO_BAR = 1e-5
# Line style of the reference lines marking the target pressure and mean period
KW_REF_LINE = {"color": "k", "lw": 1, "ls": "--", "dash_capstyle": "round"}


@dataclass
class PlotOpts:
    """Options controlling what is plotted and how."""

    diag_key: str
    target_pressure: float
    n_days_mean: float
    sim_days: float | None
    method_plt: str
    n_contours: int
    cmap: str
    norm_kind: str
    vmin: float | None
    vmax: float | None
    vcenter: float
    do_stream: bool
    linear_pressure: bool
    outdir: Path | None


def load_dataset(exp_key):
    """Load the lat-lon diagnostic files of one experiment."""
    exp = EXPERIMENTS[exp_key]
    exp_dir = paths.data_work / "lfric" / exp.group / exp.label
    files = sorted(exp_dir.glob(LATLON_GLOB))
    if not files:
        msg = f"No files matching {LATLON_GLOB} in {exp_dir}"
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


def make_norm(opts):
    """Colour normalisation, either centred on `vcenter` or plain linear."""
    if opts.norm_kind == "linear":
        return mcol.Normalize(vmin=opts.vmin, vmax=opts.vmax)
    if opts.vmin is None and opts.vmax is None:
        return mcol.CenteredNorm(vcenter=opts.vcenter)
    return mcol.TwoSlopeNorm(vcenter=opts.vcenter, vmin=opts.vmin, vmax=opts.vmax)


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
    pres_interp_var = pres.interpolate(
        [(lfric.z, var.coord(lfric.z).points)],
        iris.analysis.Linear(),  # type: ignore
    )
    var_plev = interp_cube_from_height_to_pressure_levels(
        var, pres_interp_var, [opts.target_pressure]
    )

    if opts.do_stream:
        pres_interp_uv = pres.interpolate(
            [(lfric.z, u.coord(lfric.z).points)],
            iris.analysis.Linear(),  # type: ignore
        )
        u_plev = interp_cube_from_height_to_pressure_levels(
            u, pres_interp_uv, [opts.target_pressure]
        )
        v_plev = interp_cube_from_height_to_pressure_levels(
            v, pres_interp_uv, [opts.target_pressure]
        )
    # Vertical coordinate of the pressure-axis panels: the time and zonal
    # mean pressure on the (z, y) grid, used as such for the cross-section
    # and sliced at the equator for the time-height panel.
    pres_zm_tm = last_n_day_mean(zonal_mean(pres_interp_var), opts.n_days_mean)  # type: ignore

    lats = var.coord(lfric.y).points
    lons = var.coord(lfric.x).points
    days = get_cube_rel_days(var)
    pres_bar_zy = pres_zm_tm.data * PA_TO_BAR
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
    kw_plt = {"norm": make_norm(opts), "cmap": opts.cmap, "rasterized": True}

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    axd = fig.subplot_mosaic(
        [["z-y", "t-z", "cbar"], ["y-x", "t-y", "cbar"]],
        gridspec_kw={"width_ratios": [1, 1, 0.02]},
    )

    ax = axd["z-y"]
    ax.set_title(f"{opts.n_days_mean:.0f}-day and zonal mean")
    setup_latitude_axis(ax, which="x")
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
    ax.axhline(target_bar, **KW_REF_LINE)

    ax = axd["t-z"]
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
    ax.axvspan(
        0,
        sim_days - opts.n_days_mean,
        color="k",
        alpha=0.1,
    )

    ax = axd["y-x"]
    ax.set_title(f"{opts.n_days_mean:.0f}-day mean, {target_bar:>5.2f} bar")
    setup_latitude_axis(ax)
    ax.set_xlabel("Longitude [deg]")
    ax.set_xlim(0, 360)
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
    ax.set_title(f"Zonal mean, {target_bar:>5.2f} bar")
    setup_latitude_axis(ax)
    ax.set_xlabel("Time [days]")
    ax.set_xlim(0, sim_days)
    im = getattr(ax, opts.method_plt)(days, lats, zonal_mean(var_plev).data.T, **kw_plt)
    ax.axvspan(
        0,
        sim_days - opts.n_days_mean,
        color="k",
        alpha=0.1,
    )

    fig.colorbar(im, cax=axd["cbar"], orientation="vertical")
    fig.suptitle(
        "\n".join([
            f"{exp.title} | LFRic-Atmosphere with {CATEGORIES[exp.category].title}",
            f"{var.name()} / {var.units}",
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
    "-c",
    "--category",
    "category_keys",
    type=click.Choice(sorted(CATEGORIES)),
    multiple=True,
    help="Run for all experiments in this category; repeatable.",
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
    default=300.0,
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
@click.option("--cmap", default="RdBu_r", help="Colour map.")
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
    "-o",
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Default: src/figures/drafts/EXP/.",
)
def main(exp_keys, group_keys, category_keys, **kwargs):
    """
    Plot a four-panel quick-look overview of one LFRic diagnostic.

    Experiments may be selected individually with -e, or in bulk by group
    (-g) or category (-c); the three options combine and duplicates are
    dropped.

    \b
    Examples:
      quicklook.py -e hd209_rand_t100k_nf20_tau1e5
      quicklook.py -e shj_c48_l32 -d temp -p 10000 --norm linear --cmap magma
      quicklook.py -g shj -g dhj -d temp
      quicklook.py -c rt -d temp
    """
    all_exp_keys = dict.fromkeys(exp_keys)
    for group_key in dict.fromkeys(group_keys):
        all_exp_keys.update(dict.fromkeys(GROUPS[group_key].simulations))
    for category_key in dict.fromkeys(category_keys):
        all_exp_keys.update(dict.fromkeys(CATEGORIES[category_key].simulations))
    if not all_exp_keys:
        msg = "Specify at least one experiment (-e), group (-g), or category (-c)."
        raise click.UsageError(msg)

    opts = PlotOpts(**kwargs)
    for exp_key in all_exp_keys:
        dset = load_dataset(exp_key)
        fig = plot_quicklook(dset, exp_key, opts)
        outdir = opts.outdir or paths.drafts  #  / exp_key
        figsave(fig, outdir / f"{exp_key}_{opts.diag_key}_quicklook")
        plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iris.FUTURE.date_microseconds = True
        main()
