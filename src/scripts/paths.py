"""Common paths for manipulating datasets and generating figures."""

from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src = root / "src"

# Absolute path to the `src/static` folder (contains static images)
static = src / "static"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"

# Absolute path to the `src/figures` folder (contains figure output)
figures = src / "figures"

# Constants
const = scripts / "const"

# Absolute path to the `src/data` folder (contains datasets)
data_work = src / "data"
# data = root.parent / "data"

# Ancillary files
ancil = data_work / "ancil"

# Common Met Office data
big_data = Path("/common/lfric/data")  # TODO: change
mo_ancil = big_data / "ancils"
