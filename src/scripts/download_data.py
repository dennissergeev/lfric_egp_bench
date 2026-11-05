#!/usr/bin/env python
"""Download LFRic output from a remote host via rsync."""

import subprocess

import click
import paths
from common import EXPERIMENTS

REMOTE_CYLC_RUN = {
    "monsoonhpc": "/home/users/denis.sergeev.ext/cylc-run",
}


@click.command()
@click.argument("exp_keys", nargs=-1, type=click.Choice(sorted(EXPERIMENTS)))
@click.option(
    "--host",
    type=click.Choice(sorted(REMOTE_CYLC_RUN)),
    default="monsoonhpc",
    show_default=True,
    help="Remote host to download the data from.",
)
def main(exp_keys, host):
    """Download LFRic output for EXP_KEYS (default: all experiments)."""
    exps = [EXPERIMENTS[key] for key in exp_keys] or EXPERIMENTS.values()
    group_labels = {(exp.group, exp.label) for exp in exps}
    remote_cylc_run = REMOTE_CYLC_RUN[host]
    for group, label in sorted(group_labels):
        local_dir = paths.data_work / "lfric" / group / label
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_exp = f"{host}:{remote_cylc_run}/{group}/{label}"
        # NetCDF output has unique names per cycle, so it is stored flat.
        output = [f"{remote_exp}/work/20*/lfric_atm/lfric_*.nc"]
        # These names repeat in every cycle, so keep work/<cycle>/lfric_atm/ locally
        # (--relative recreates the path after the "/./" marker).
        cycle_files = [
            f"{remote_exp}/./work/20*/lfric_atm/{name}"
            for name in ("configuration.nml",)  # "PET*.Log")
        ]
        for flags, sources in (("-amPvz", output), ("-amPvzR", cycle_files)):
            try:
                subprocess.run(["rsync", flags, *sources, f"{local_dir}/"], check=True)
            except subprocess.CalledProcessError as e:
                click.echo(
                    f"rsync failed for {group}/{label}: exit code {e.returncode}",
                    err=True,
                )


if __name__ == "__main__":
    main()
