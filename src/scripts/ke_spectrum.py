"""Kinetic energy spectrum diagnostic."""

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from spharm import Spharmt, getspecindx
from windspharm.standard import VectorWind
from windspharm.tools import order_latdim, prep_data

EARTH_RADIUS = 6_371_200.0  # metres

VRT_NAME = "vertical_vorticity"
DIV_NAME = "divergence_of_wind"
U_NAME = "u_in_w3"
V_NAME = "v_in_w3"


def _prep_pair(cube_a, cube_b):
    """
    Reshape a pair of cubes to the (nlat, nlon, nt) layout used by spharm.

    Returns the prepped arrays, the `prep_data` info dictionary and the
    latitude/longitude dimension indices of the original cubes.
    """
    lat_coord = cube_a.coord(axis="Y")
    lon_coord = cube_a.coord(axis="X")
    (lat_dim,) = cube_a.coord_dims(lat_coord)
    (lon_dim,) = cube_a.coord_dims(lon_coord)

    dim_chars = []
    for dim in range(cube_a.ndim):
        if dim == lat_dim:
            dim_chars.append("y")
        elif dim == lon_dim:
            dim_chars.append("x")
        else:
            dim_chars.append(chr(ord("a") + dim))
    dimorder = "".join(dim_chars)

    a_prep, info = prep_data(np.asarray(cube_a.data, dtype=np.float32), dimorder)
    b_prep, _ = prep_data(np.asarray(cube_b.data, dtype=np.float32), dimorder)
    _, a_prep, b_prep = order_latdim(lat_coord.points, a_prep, b_prep)
    return a_prep, b_prep, info, lat_dim, lon_dim


def _spec_to_ke_cube(vrtspec, divspec, ntrunc, rsphere, info, cube, lat_dim, lon_dim):
    """Combine spectral vorticity and divergence into a KE spectrum cube."""
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

    other_dims = [d for d in range(cube.ndim) if d not in (lat_dim, lon_dim)]
    dim_coords_and_dims = []
    for new_dim, orig_dim in enumerate(other_dims):
        for coord in cube.coords(dim_coords=True):
            if cube.coord_dims(coord) == (orig_dim,):
                dim_coords_and_dims.append((coord.copy(), new_dim))
    wavenumber_coord = DimCoord(n, long_name="wavenumber", units="1")
    dim_coords_and_dims.append((wavenumber_coord, ke_n.ndim - 1))

    return Cube(
        ke_n,
        long_name="kinetic_energy_spectrum",
        units="m2 s-2",
        dim_coords_and_dims=dim_coords_and_dims,
    )


def ke_spectrum_from_uv(u_cube, v_cube, gridtype="regular", rsphere=EARTH_RADIUS):
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

    See Also
    --------
    ke_spectrum_from_vrtdiv, ke_spectrum_from_cubelist
    """
    u_prep, v_prep, info, lat_dim, lon_dim = _prep_pair(u_cube, v_cube)

    vw = VectorWind(u_prep, v_prep, gridtype=gridtype, rsphere=rsphere)
    ntrunc = vw.s.nlat - 1
    vrtspec, divspec = vw.s.getvrtdivspec(u_prep, v_prep)

    return _spec_to_ke_cube(
        vrtspec, divspec, ntrunc, rsphere, info, u_cube, lat_dim, lon_dim
    )


def ke_spectrum_from_vrtdiv(
    vrt_cube, div_cube, gridtype="regular", rsphere=EARTH_RADIUS
):
    """
    Compute the kinetic energy spectrum from vorticity and divergence.

    Equivalent to `ke_spectrum_from_uv`, but starting from the relative vertical
    vorticity and horizontal divergence fields instead of the wind
    components: the two fields are transformed to spherical harmonics
    directly, skipping the vector wind decomposition. Useful when the
    model output contains vorticity and divergence but not (or not on
    the same grid as) the wind components.

    Parameters
    ----------
    vrt_cube, div_cube: iris.cube.Cube
        Relative vertical vorticity and horizontal divergence of the
        wind (both in s-1) on a common lat-lon grid, with dimensions
        such as (level_height, latitude, longitude). Latitude and
        longitude must each be a single dimension of the cubes; any
        other dimensions are preserved in the output.
    gridtype: str, optional
        Either "regular" (default) for an evenly-spaced latitude grid,
        or "gaussian".
    rsphere: float, optional
        Planetary radius in metres, used to scale the spectrum.

    Returns
    -------
    iris.cube.Cube
        Kinetic energy spectrum in m2 s-2, with a "wavenumber" dimension
        (n = 1, 2, ...) replacing latitude and longitude.
    """
    vrt_prep, div_prep, info, lat_dim, lon_dim = _prep_pair(vrt_cube, div_cube)

    nlat, nlon = vrt_prep.shape[:2]
    sph = Spharmt(nlon, nlat, gridtype=gridtype, rsphere=rsphere)
    ntrunc = nlat - 1
    vrtspec = sph.grdtospec(vrt_prep, ntrunc)
    divspec = sph.grdtospec(div_prep, ntrunc)

    return _spec_to_ke_cube(
        vrtspec, divspec, ntrunc, rsphere, info, vrt_cube, lat_dim, lon_dim
    )


def ke_spectrum_from_cubelist(
    cubes, source="auto", gridtype="regular", rsphere=EARTH_RADIUS, **names
):
    """
    Compute the kinetic energy spectrum from a cube list.

    Parameters
    ----------
    cubes: iris.cube.CubeList
        Cube list containing either the wind components or the
        vorticity and divergence fields (or both).
    source: str, optional
        "wind" to use the wind components, "vrtdiv" to use vorticity and
        divergence, or "auto" (default) to prefer vorticity and
        divergence if both pairs are present.
    gridtype, rsphere:
        Passed through to `ke_spectrum_from_uv` / `ke_spectrum_from_vrtdiv`.
    **names:
        Optional overrides for the cube names to extract:
        `vrt_name`, `div_name`, `u_name`, `v_name`.

    Returns
    -------
    iris.cube.Cube
        Kinetic energy spectrum in m2 s-2.
    """
    vrt_name = names.pop("vrt_name", VRT_NAME)
    div_name = names.pop("div_name", DIV_NAME)
    u_name = names.pop("u_name", U_NAME)
    v_name = names.pop("v_name", V_NAME)
    if names:
        raise TypeError(f"Unexpected keyword arguments: {sorted(names)}")

    available = {cube.name() for cube in cubes}
    has_vrtdiv = {vrt_name, div_name} <= available
    has_wind = {u_name, v_name} <= available

    if source == "auto":
        source = "vrtdiv" if has_vrtdiv else "wind"

    if source == "vrtdiv":
        if not has_vrtdiv:
            raise ValueError(
                f"Cube list does not contain {vrt_name!r} and {div_name!r}"
            )
        return ke_spectrum_from_vrtdiv(
            cubes.extract_cube(vrt_name),
            cubes.extract_cube(div_name),
            gridtype=gridtype,
            rsphere=rsphere,
        )
    if source == "wind":
        if not has_wind:
            raise ValueError(f"Cube list does not contain {u_name!r} and {v_name!r}")
        return ke_spectrum_from_uv(
            cubes.extract_cube(u_name),
            cubes.extract_cube(v_name),
            gridtype=gridtype,
            rsphere=rsphere,
        )
    raise ValueError(f"Unknown source: {source!r}")
