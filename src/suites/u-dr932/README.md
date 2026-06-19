u-dr932
=======

This is a standalone LFRic suite for Shallow and Deep Hot Jupiter Temperature Forcing Cases.

Since Apps vn3.0 the workflow has been setup to run using git sources. Please
see below for instructions on modifying suites to run with git sources. All
revisions of the LFRic trunks that were available on MOSRS are also available
from git and so this setup of suite can be used for older LFRic versions. Any
MOSRS branches will need porting to git before being used.

Setting Source Codes
--------------------

Different sources for the required source codes can be set in the
`dependencies.yaml` file. LFRic Apps requires sources to be set for Casim,
Jules, LFRic Apps, LFRic Core, Socrates and UKCA. But other git sources can be
added if data from them is required, eg. the socrates-spectral repository.

Multiple branches can be combined and merged if desired (similar to the
fcm_extract functionality for working with subversion repositories). To do this,
set multiple sources in the `dependencies.yaml` file:

```yaml
lfric_apps:
    - source: git@github.com:MetOffice/lfric_apps.git
      ref: 2025.12.1
    - source: git@github.com:my_fork/lfric_apps.git
      ref: my_branch
```

This will first clone the MetOffice Apps repository and checkout the `2025.12.1`
tag. It will then merge in my_branch from my_fork. It is up to the user to
ensure the branches are mergeable - an error will be raised if conflicts occur (
although conflicts in the rose-stem directory and repository dependencies.yaml
file will be ignored).

Changing a workflow from fcm to git sources
-------------------------------------------

To change a workflow to work with git sources will involve some manual changes.

* Take a copy of the `app/extract` directory from here.
* Copy the file extracts for `get_git_sources.py` and `merge_sources.py` as well
  as the template variables `MIRROR_LOC`, `USE_MIRRORS` and `USE_TOKENS` from
  the `rose-suite.conf`. You may also need to update the tweak_iodef source.
* In the `flow.cylc`:
  * add the new `extract` task. See the `graph` section and the `[[extract]]`
    sections.
  * Add `PHYSICS_ROOT = $SOURCE_ROOT` to the environment variable list.
  * Add `install = dependencies.yaml` to the `scheduler` section.
* Create a `dependencies.yaml` file - see above.

By default the workflow uses local git mirrors. Directly cloning from github can
also be done if correct authentication has been setup by setting
`-S USE_MIRROR=false`.

Using the mirrors requires either setting the `MIRROR_LOC` variable or setting
up the `localmirrors:` alias by running,
`git config --global url."hostname:/path/to/git_mirrors/".insteadOf "localmirrors:"`

