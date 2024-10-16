import platform
from pathlib import Path
import lsdo_geo as lg
import lsdo_function_spaces as lfs
import csdl_alpha as csdl


def import_geometry(
    file_name: str,
    file_path: Path,
    refit: bool = False,
    refit_num_coefficients: tuple = (40, 40),
    refit_b_spline_order: tuple = (4, 4),
    refit_resolution: tuple = (200, 200),
    rotate_to_body_fixed_frame: bool = True,
    scale: float = 1.0
) -> lg.Geometry:
    """
    Import a (OpenVSP) .stp file.

    Parameters
    ----------
    file_name : str
        Name of the file (.stp support only at this time).
    file_path : Path
        Path to the file.
    refit : bool, optional
        Whether to refit the geometry, by default False.
    refit_num_coefficients : tuple, optional
        Number of coefficients for refitting, by default (40, 40).
    refit_b_spline_order : tuple, optional
        Order of the B-spline refit, by default (4, 4).
    refit_resolution : tuple, optional
        Resolution for the refit, by default (200, 200).
    rotate_to_body_fixed_frame : bool, optional
        Apply a 180° rotation about the z-axis followed by a 180° rotation about the x-axis, by default True.
    scale : float, optional
        Scaling factor for the geometry, by default 1.0.

    Returns
    -------
    lg.Geometry
        The imported geometry.

    Raises
    ------
    Exception
        If the file is not in .stp format.
    Exception
        If the file path is unknown or the file does not exist.
    """

    # Check the operating system and set num_workers to 1 if not Linux
    if platform.system() != "Linux":
        lfs.num_workers = 1

    # Validate file extension
    if not file_name.endswith(".stp"):
        raise Exception(f"Can only import '.stp' files at the moment. Received {file_name}")

    # Validate file path
    full_path = file_path / file_name
    if not full_path.is_file():
        raise Exception(f"Unknown file path or file. File path: {full_path}")

    # Import geometry
    geometry = lg.import_geometry(full_path, parallelize=False, scale=scale)

    # Refit geometry if required
    if refit:
        refit_space = lfs.BSplineSpace(2, refit_b_spline_order, refit_num_coefficients)
        geometry.refit(refit_space, grid_resolution=refit_resolution)

    # Rotate to body-fixed frame if required
    if rotate_to_body_fixed_frame:
        for function in geometry.functions.values():
            coeffs = function.coefficients.value
            coeffs[:, :, 0] = -coeffs[:, :, 0]
            coeffs[:, :, 2] = -coeffs[:, :, 2]
            function.coefficients = csdl.Variable(value=coeffs)

    return geometry