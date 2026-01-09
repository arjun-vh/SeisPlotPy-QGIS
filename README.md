# SeisPlotPy (QGIS Plugin)

**SeisPlotPy** is a subsurface acoustic data visualization and interpretation plugin for **QGIS**.  
It allows users to load, view, navigate, analyse and interpret **2D SEG-Y (post-stack) seismic data** directly inside QGIS.

SeisPlotPy is designed for earth scientists, researchers, students, and geophysics workflows involving seismic reflection or sub-bottom profiler data.

---

## Features

- Load and visualize **SEG-Y** files (via `segyio`)
- Interactive controls for:
- Various Colormaps (e.g., seismic, grayscale, etc.)
- Linked **navigation with QGIS map canvas**
- Horizon picking and editing
- Export figures for publications and reports
- Supports **pyqtgraph** for fast seismic rendering
- Built with PyQt and integrates naturally with QGIS UI

---

## Installation

### 1️⃣ Install dependencies

Before enabling the plugin, install the required Python packages into the **QGIS Python environment**:

python -m pip install segyio numpy scipy pandas matplotlib pyqtgraph

Tip: Users may also install these using the **QGIS Pip Manager** plugin.

---

### 2️⃣ Install SeisPlotPy Plugin

#### Option A — Official QGIS Plugin Repository (Recommended)
This is the easiest way to stay up to date with the latest features. Since SeisPlotPy is currently in active development, you may need to enable experimental versions.

1. Open QGIS and go to Plugins → Manage and Install Plugins...
2. Click on the Settings tab on the left sidebar.
3. Check the box that says "Show also experimental plugins".
4. Switch to the All tab and search for SeisPlotPy.
5. Select the plugin and click Install Plugin.

#### Option B — From ZIP 

1. Download the latest release ZIP from the repository:
    https://github.com/arjun-vh/SeisPlotPy-QGIS/releases
2. Open QGIS → `Plugins → Install from ZIP`
3. Select the downloaded file and install
4. Restart QGIS

#### Option C — Manual Installation

Copy the `SeisPlotPy` folder into:

QGIS profile folder → python/plugins/


Then enable it under:  
`Plugins → Manage and Install Plugins → Installed → SeisPlotPy`

---

##  Supported Data

| Format | Supported | Notes |
|--------|-----------|-------|
| SEG-Y | ✅ Yes | Preferred and tested |
| Sub-bottom profiler data exports | ⚠ Yes (if SEG-Y structured) |

---

## Dependencies

| Library | Required |
|---------|----------|
| segyio | ✔ |
| numpy | ✔ |
| scipy | ✔ |
| pandas | ✔ |
| matplotlib | ✔ |
| pyqtgraph | ✔ |

The plugin will notify users if any dependencies are missing.

---

## Compatibility

| Environment | Status |
|-------------|--------|
| QGIS 3.16+ | ✔ tested |
| Windows | ✔ tested |
| Linux | ⚠ expected to work (not yet verified) |
| macOS | ⚠ dependent on Python env setup |

---

## License

SeisPlotPy is released under the **GPL-2.0 license**.

---

## Issues & Feature Requests

Found a bug? Need a feature?  
Please report it here:

https://github.com/arjun-vh/SeisPlotPy-QGIS/issues

## Citation

If you use SeisPlotPy in your research, reports, or professional work, please acknowledge the software using the following citation:

Arjun, V.H. (2025). arjun-vh/SeisPlotPy-QGIS: SeisPlotPy QGIS Plugin v0.9.5. Zenodo. https://doi.org/10.5281/zenodo.17960132





