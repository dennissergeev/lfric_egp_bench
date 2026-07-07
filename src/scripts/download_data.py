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
        remote_glob = f"{remote_cylc_run}/{group}/{label}/work/20*/lfric_atm/lfric_*.nc"
        subprocess.run(
            ["rsync", "-amPvz", f"{host}:{remote_glob}", f"{local_dir}/"],
            check=True,
        )


if __name__ == "__main__":
    main()
