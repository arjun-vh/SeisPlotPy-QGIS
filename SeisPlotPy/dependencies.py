import importlib
from qgis.PyQt.QtWidgets import QMessageBox

# Map import names to the package names users should install
REQUIRED_MODULES = {
    "segyio": "segyio",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pyqtgraph": "pyqtgraph",
    "matplotlib": "matplotlib",
}


def check_dependencies(iface):
    """
    Check that all required Python packages are importable.

    Returns:
        True  -> all good, plugin can load
        False -> something missing, message shown to user, plugin should not load
    """
    missing = []

    for import_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return True

    # Build strings for display (comma) and command (space)
    missing_list = ", ".join(missing)
    missing_cmd  = " ".join(missing)

    msg = (
        "The 'SeisPlotPy' plugin requires the following Python packages in the "
        "QGIS Python environment:\n\n"
        f"    {missing_list}\n\n"
        "They are currently NOT available.\n\n"
        "▶ On Windows (QGIS / OSGeo4W):\n"
        "  1. Close QGIS completely.\n"
        "  2. Open the 'OSGeo4W Shell' from the Start Menu.\n"
        "  3. Run the command (all on one line):\n"
        f"     python -m pip install {missing_cmd}\n\n"
        "▶ On Linux / macOS:\n"
        "  Install them into the Python environment that QGIS uses, for example:\n"
        f"     python3 -m pip install {missing_cmd}\n\n"
        "▶ Alternative (any OS):\n"
        "  You may also use the 'QGIS Pip Manager' plugin to install these\n"
        "  packages inside the QGIS environment.\n\n"
        "After installation, restart QGIS and enable SeisPlotPy again."
    )

    QMessageBox.critical(
        iface.mainWindow(),
        "SeisPlotPy - Missing Python Dependencies",
        msg,
    )

    return False
