# Getting Started

## 1. Requirements & Installation

SeisPlotPy relies on several standard scientific Python libraries. Ensure these are installed in your QGIS Python environment prior to enabling the plugin.

**Required Packages:**
- `segyio` (>= 1.9.0)
- `numpy` (>= 1.21.0)
- `scipy` (>= 1.10.0)
- `pandas` (>= 1.3.0)
- `pyqtgraph` (>= 0.13.0)
- `matplotlib` (>= 3.5.0)
- `markdown` (>= 3.3.0)

When you first launch SeisPlotPy, it will perform a dependency check. If any packages are missing, a dialog will appear listing exactly which ones you need to install.

### Installation Methods

#### Option A — Official QGIS Plugin Repository (Recommended)
1. Open QGIS and go to **Plugins > Manage and Install Plugins...**
2. Click on the **Settings** tab on the left sidebar.
3. Switch to the **All** tab and search for *SeisPlotPy*.
4. Select the plugin and click **Install Plugin**.

#### Option B — From ZIP
1. Download the latest release ZIP from the [GitHub releases page](https://github.com/arjun-vh/SeisPlotPy-QGIS/releases).
2. Open QGIS → **Plugins > Install from ZIP**.
3. Select the downloaded file and install.

---

## 2. Window Anatomy

When you open a SEG-Y file, SeisPlotPy launches its own dedicated viewer window. This window operates independently of the main QGIS interface, allowing you to move it to a second monitor.

The window is split into two main sections:

### The Left Sidebar (Controls)
This panel contains all your immediate tools:
- **Load buttons:** `Load Single SEG-Y` and `Batch load multiple SEGY`.
- **Active Viewport:** Controls to zoom into specific trace ranges, change the X-Axis reference (e.g., from Trace Index to CDP), and control data decimation.
- **Visualization:** Dropdowns for Colormap, a Contrast percent spinbox, rendering toggles (Smooth, High Res, Grid, Flip X, Domain), and a toggle for the Color Legend.
- **Export:** Controls to set the precise width and height (in inches) of your exported figure, and an `Export Figure` button.
- **Status Label:** A readout at the very bottom confirming actions like "Applied AGC" or "Horizons auto-saved ✓".

### The Right Plot Area (Viewer)
This is the high-performance `pyqtgraph` canvas where your seismic data is rendered. 
- You can pan by left-click dragging, and zoom by right-click dragging or using the scroll wheel.
- **Live Status Bar:** Look at the bottom-right corner of the window while hovering over the plot to see real-time coordinate readouts, including seismic amplitude, mapped CRS coordinates, and projected Lat/Lon.

---

## 3. Menu Bar Reference

At the top of the SeisPlotPy window is a menu bar with advanced tools:

| Menu | Contents |
|------|----------|
| **File** | Load SEG-Y, Export PDF/PNG, Export SEG-Y Subset... |
| **Processing** | Apply AGC, Bandpass Filter, Reset to Raw Data |
| **Attributes** | Instantaneous Amplitude (Envelope), Phase, Cosine Phase, Frequency, RMS Amplitude |
| **Tools** | Setup Geometry / Distance, Horizon Manager, Fault Manager, Header Utilities, Frequency Spectrum, Amplitude Histogram |
| **Help** | Documentation, About SeisPlotPy |

---

## 4. QGIS Toolbar

In the main QGIS window, the plugin adds a small toolbar with two critical buttons:
1. **SeisPlotPy Icon:** Opens a new, blank viewer window. (You can open multiple windows to view different lines simultaneously).
2. **Link/Nav Tool (Crosshair):** A toggle button. You must **click this button** to activate map canvas linking. When active, hovering over the seismic line on your QGIS map will update the SeisPlotPy viewer.
