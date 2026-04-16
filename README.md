# SeisPlotPy - QGIS Plugin

SeisPlotPy is an advanced, open-source QGIS plugin designed for the visualization, navigation, and interpretation of SEG-Y seismic data directly within a geographic information system environment. 

By bridging the gap between high-performance geophysical data arrays and spatial mapping, SeisPlotPy allows geoscientists to dynamically link seismic traces to real-world coordinates, pick horizons, and apply real-time signal processing.

## 🌟 Key Features

* **Interactive Visualization:** Fast, high-resolution rendering of SEG-Y files using `pyqtgraph`. Includes customizable contrast, color palettes, and High-Res interpolation.
* **Spatial Navigation:** Automatically extracts CDP/Source/Group coordinates, computes cumulative distances, and generates a dynamic QGIS vector layer. Double-clicking the map opens the seismic viewer; hovering over the map highlights the exact trace.
* **Interpretation & Flattening:** Pick, edit, and manage horizons and faults. Features a dynamic "Flatten" mode to warp the seismic image and other interpretations to a reference horizon.
* **Real-Time Processing:** Apply AGC, Bandpass filtering, and instantaneous attributes (Envelope, Phase, Frequency, RMS Amplitude) on the fly.
* **Header Utilities:** Inspect binary and trace headers, generate QC plots, view amplitude histograms, and perform bulk CSV header patching/exporting.

## 📦 Installation & Dependencies

SeisPlotPy relies on several standard scientific Python libraries. Ensure these are installed in your QGIS Python environment prior to enabling the plugin.

**Required Packages:**
* `segyio`
* `numpy`
* `scipy`
* `pandas`
* `pyqtgraph`
* `matplotlib`

### 1️⃣ Install dependencies

Before enabling the plugin, install the required Python packages into the **QGIS Python environment**:

python -m pip install segyio numpy scipy pandas matplotlib pyqtgraph

Tip: Users may also install these using the **QGIS Pip Manager** plugin.

---

The plugin will notify users if any dependencies are missing.

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

## 🚀 Quick Start Guide

1. **Launch the Plugin:** Click the SeisPlotPy icon in the QGIS toolbar.
2. **Load Data:** Click **Load SEG-Y** and select your file. The tool will automatically parse the headers and render the first subset of traces.
3. **Map the Line:** Go to **Tools > Setup Geometry / Distance**. Select your X/Y coordinate headers (e.g., `CDP_X`, `CDP_Y`) and your desired Coordinate Reference System (CRS). The plugin will calculate cumulative distance and draw the seismic line on your QGIS map canvas.
4. **Navigate:** Use the SeisPlotPy map tool in QGIS to hover over the line and see the corresponding trace in the viewer. 
5. **Interpret:** Open **Tools > Horizon Manager**, add a new horizon, toggle "Pick," and left-click on the seismic plot to start mapping reflectors.

## 📖 Comprehensive Documentation

For detailed instructions on specific modules, please refer to our Wiki / Docs:

1. [Navigation & Spatial Linking](docs/navigation.md)
2. [Display Controls & High-Res Rendering](docs/display.md)
3. [Seismic Processing & Attributes](docs/processing.md)
4. [Horizon & Fault Interpretation](docs/interpretation.md)
5. [Header Exploration & Patching](docs/headers.md)
6. [Exporting Data & Figures](docs/exporting.md)

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

## 🤝 Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on GitHub.

https://github.com/arjun-vh/SeisPlotPy-QGIS/issues

## Citation

If you use SeisPlotPy in your research, reports, or professional work, please acknowledge the software using the following citation:

Arjun, V.H. (2025). arjun-vh/SeisPlotPy-QGIS: SeisPlotPy QGIS Plugin v0.9.5. Zenodo. https://doi.org/10.5281/zenodo.17960131





