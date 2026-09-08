"""Kinetic energy spectrum diagnostic."""

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from spharm import getspecindx
from windspharm.standard import VectorWind
from windspharm.tools import order_latdim, prep_data

EARTH_RADIUS = 6_371_200.0  # metres


def ke_spectrum(u_cube, v_cube, gridtype="regular", rsphere=EARTH_RADIUS):
    """
    Compute the kinetic energy spectrum as a function of total wavenumber.

    The horizontal wind is decomposed into spherical harmonics and the
    kinetic energy per total (meridional) wavenumber `n` is computed
    from the spectral vorticity and divergence coefficients, following
    Boer (1983) and Koshyk & Hamilton (2001). Summing the returned
    spectrum over wavenumber recovers the global-mean kinetic energy
    of the horizontal wind.

    Parameters
    ----------
    u_cube, v_cube: iris.cube.Cube
        Eastward and northward wind components on a common lat-lon grid,
        with dimensions such as (level_height, latitude, longitude).
        Latitude and longitude must each be a single dimension of the
        cubes; any other dimensions (e.g. level_height, time) are
        preserved in the output.
    gridtype: str, optional
        Either "regular" (default) for an evenly-spaced latitude grid,
        or "gaussian".
    rsphere: float, optional
        Planetary radius in metres, used to scale the spectrum. Defaults
        to Earth's radius; for the exoplanet experiments in this project
        pass e.g. ``const.radius.data.item()`` from the relevant
        `Experiment.const`.

    Returns
    -------
    iris.cube.Cube
        Kinetic energy spectrum in m2 s-2, with a "wavenumber" dimension
        (n = 1, 2, ...) replacing latitude and longitude.
    """
    lat_coord = u_cube.coord(axis="Y")
    lon_coord = u_cube.coord(axis="X")
    (lat_dim,) = u_cube.coord_dims(lat_coord)
    (lon_dim,) = u_cube.coord_dims(lon_coord)

    dim_chars = []
    for dim in range(u_cube.ndim):
        if dim == lat_dim:
            dim_chars.append("y")
        elif dim == lon_dim:
            dim_chars.append("x")
        else:
            dim_chars.append(chr(ord("a") + dim))
    dimorder = "".join(dim_chars)

    u_prep, info = prep_data(u_cube.data, dimorder)
    v_prep, _ = prep_data(v_cube.data, dimorder)
    _, u_prep, v_prep = order_latdim(lat_coord.points, u_prep, v_prep)

    vw = VectorWind(u_prep, v_prep, gridtype=gridtype, rsphere=rsphere)
    ntrunc = vw.s.nlat - 1
    vrtspec, divspec = vw.s.getvrtdivspec(u_prep, v_prep)
    indxm, indxn = getspecindx(ntrunc)

    # Sum |coeff|^2 over the full range m = -n..n. Only m >= 0 is stored,
    # so contributions from m > 0 are doubled to account for m < 0.
    doubling = np.where(indxm == 0, 1.0, 2.0)[:, np.newaxis]
    power = (np.abs(vrtspec) ** 2 + np.abs(divspec) ** 2) * doubling

    ke_n = np.zeros((ntrunc + 1,) + power.shape[1:])
    np.add.at(ke_n, indxn, power)
    n = np.arange(1, ntrunc + 1)
    ke_n = 0.25 * rsphere**2 * ke_n[1:] / (n * (n + 1))[:, np.newaxis]

    # Move the wavenumber axis (currently first) to last, and restore the
    # shape/order of the non-lat-lon ("other") dimensions.
    other_shape = info["intermediate_shape"][2:]
    ke_n = np.moveaxis(ke_n.reshape((ntrunc,) + other_shape), 0, -1)  # type: ignore

    other_dims = [d for d in range(u_cube.ndim) if d not in (lat_dim, lon_dim)]
    dim_coords_and_dims = []
    for new_dim, orig_dim in enumerate(other_dims):
        for coord in u_cube.coords(dim_coords=True):
            if u_cube.coord_dims(coord) == (orig_dim,):
                dim_coords_and_dims.append((coord.copy(), new_dim))
    wavenumber_coord = DimCoord(n, long_name="wavenumber", units="1")
    dim_coords_and_dims.append((wavenumber_coord, ke_n.ndim - 1))

    return Cube(
        ke_n,
        long_name="kinetic_energy_spectrum",
        units="m2 s-2",
        dim_coords_and_dims=dim_coords_and_dims,
    )
