# SeisPlotPy (QGIS Plugin)

**SeisPlotPy** is a acoustic subsurface data visualization and interpretation plugin for **QGIS**.  
It allows users to load, view, navigate, and interpret **SEG-Y (post-stack) seismic data** directly inside QGIS.

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

#### Option A — From ZIP (recommended for testing)

1. Download the latest release ZIP from the repository:
    https://github.com/arjun-vh/SeisPlotPy-QGIS/releases
2. Open QGIS → `Plugins → Install from ZIP`
3. Select the downloaded file and install
4. Restart QGIS

#### Option B — Manual Installation

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

## 🧾 License

SeisPlotPy is released under the **GPL-2.0 license**.

---

## Issues & Feature Requests

Found a bug? Need a feature?  
Please report it here:

https://github.com/arjun-vh/SeisPlotPy-QGIS/issues





