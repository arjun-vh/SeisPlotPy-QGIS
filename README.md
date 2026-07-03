# SeisPlotPy - QGIS Plugin

SeisPlotPy is an advanced, open-source QGIS plugin designed for the visualization, navigation, and interpretation of 2D SEG-Y seismic data directly within the QGIS environment. 

By bridging the gap between high-performance geophysical data arrays and spatial mapping, SeisPlotPy allows geoscientists to dynamically link seismic traces to real-world coordinates, pick horizons, and apply real-time signal processing.

## 🌟 Key Features

* **Interactive Visualization:** Fast, high-resolution rendering of SEG-Y files using `pyqtgraph`. Includes customizable contrast, color palettes, and High-Res interpolation.
* **QGIS Project Persistence:** Save and restore all open seismic viewer windows, display settings, and navigation layers directly within a standard QGIS `.qgz` project file.
* **Batch SEG-Y Loading:** Map an entire 2D seismic survey grid to the QGIS canvas in one background operation.
* **Spatial Navigation:** Automatically extracts CDP/Source/Group coordinates, computes cumulative distances, and generates a dynamic QGIS vector layer. Double-clicking the map opens the seismic viewer; hovering over the map highlights the exact trace.
* **Live Coordinate Readout:** Real-time display of trace position, seismic amplitude, CRS coordinates, and WGS84 Lat/Lon in the status bar while hovering.
* **Interpretation & Flattening:** Pick, edit, and manage horizons and faults. Features a dynamic "Flatten" mode to warp the seismic image and other interpretations to a reference horizon.
* **Real-Time Processing:** Apply AGC, Bandpass filtering, and instantaneous attributes (Envelope, Phase, Frequency, RMS Amplitude) on the fly.
* **Header Utilities & Fallback Reader:** Inspect headers, view amplitude histograms, patch headers via CSV, and securely open non-standard files using the raw fallback reader (with automatic IBM float conversion).

## 📦 Installation & Dependencies

SeisPlotPy relies on several standard scientific Python libraries. Ensure these are installed in your QGIS Python environment prior to enabling the plugin.

**Required Packages:**
* `segyio` (>= 1.9.0)
* `numpy` (>= 1.21.0)
* `scipy` (>= 1.10.0)
* `pandas` (>= 1.3.0)
* `pyqtgraph` (>= 0.13.0)
* `matplotlib` (>= 3.5.0)
* `markdown` (>= 3.3.0)

### 1️⃣ Install dependencies

Before enabling the plugin, install the required Python packages into the **QGIS Python environment**:

`python -m pip install segyio numpy scipy pandas matplotlib pyqtgraph markdown`

Tip: Users may also install these using the **QGIS Pip Manager** plugin. The plugin will notify users via a popup dialog if any dependencies are missing upon first launch.

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
1. Download the latest release ZIP from the repository: https://github.com/arjun-vh/SeisPlotPy-QGIS/releases
2. Open QGIS → `Plugins → Install from ZIP`
3. Select the downloaded file and install
4. Restart QGIS

#### Option C — Manual Installation
Copy the `SeisPlotPy` folder into: `QGIS profile folder → python/plugins/`
Then enable it under: `Plugins → Manage and Install Plugins → Installed → SeisPlotPy`

---

## 🚀 Quick Start Guide

1. **Launch the Plugin:** Click the SeisPlotPy icon in the QGIS toolbar.
2. **Load Data:** Click **Load Single SEG-Y** and select your file. The tool will automatically parse the headers and render the first subset of traces.
3. **Map the Line:** Go to **Tools > Setup Geometry / Distance**. Select your X/Y coordinate headers (e.g., `CDP_X`, `CDP_Y`) and the Coordinate Reference System (CRS). The plugin will calculate cumulative distance and draw the seismic line on your QGIS map canvas.
4. **Navigate:** Click the **Link/Nav Tool** icon in the QGIS toolbar to activate the crosshair. Hover over the line on the map to see the corresponding trace in the viewer. 
5. **Interpret:** Open **Tools > Horizon Manager**, add a new horizon, toggle "Pick," and left-click on the seismic plot to start mapping reflectors.

## 📖 Comprehensive Documentation

For detailed instructions, refer to the integrated documentation. Within the plugin, click **Help > Documentation** to open the native e-book viewer, or read the markdown files directly:

1. [Getting Started](docs/getting_started.md)
2. [Loading Data](docs/loading.md)
3. [Display Controls & High-Res Rendering](docs/display.md)
4. [Navigation & Spatial Linking](docs/navigation.md)
5. [Seismic Processing & Attributes](docs/processing.md)
6. [Horizon & Fault Interpretation](docs/interpretation.md)
7. [Header Exploration & Patching](docs/headers.md)
8. [Exporting Data & Figures](docs/exporting.md)
9. [QGIS Project Persistence](docs/project_persistence.md)
10. [Batch SEG-Y Loading](docs/batch_loading.md)
11. [Quick Reference](docs/reference.md)

## Compatibility

| Environment | Status |
|-------------|--------|
| QGIS 3.16+ | ✔ tested |
| Windows | ✔ tested |
| Linux | ⚠ expected to work (not yet verified) |
| macOS | ⚠ dependent on Python env setup |

---

## License

SeisPlotPy is released under the **GPL-3.0 license**.

---

## 🤝 Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on GitHub:
https://github.com/arjun-vh/SeisPlotPy-QGIS/issues

## Citation

If you use SeisPlotPy in your research, reports, or professional work, please acknowledge the software using the following citation:

`Arjun, V.H. (2025). arjun-vh/SeisPlotPy-QGIS: SeisPlotPy QGIS Plugin v1.0.5. Zenodo. https://doi.org/10.5281/zenodo.17960131`
