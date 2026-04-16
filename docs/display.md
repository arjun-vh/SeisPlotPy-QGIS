# Display Controls & High-Res Rendering

SeisPlotPy is built on `pyqtgraph` to ensure extremely fast rendering of large 2D arrays. To help you highlight specific structural or stratigraphic features, the plugin offers a comprehensive suite of real-time display controls.

All display settings are located in the left-hand sidebar of the main SeisPlotPy window.

---

## 1. Color & Contrast Management

### Colormaps
The **Colormap** dropdown allows you to instantly switch the color palette of the seismic data. SeisPlotPy utilizes standard Matplotlib colormaps. 
* **Grayscale / Seis:** Standard structural viewing.
* **Seismic / RdBu:** Divergent colormaps excellent for viewing zero-phase data, where zero crossings are white, and peaks/troughs are red/blue.
* **Jet / Viridis:** Useful for viewing instantaneous attributes like Envelope or Frequency.

### Contrast (Percentile Clipping)
Instead of arbitrary slider values, SeisPlotPy uses a statistical **Percentile Clip** to manage contrast.
* The **Contrast (%)** spinbox sets the percentile of amplitude values to clip at. 
* **Example:** Setting it to `98` means the colorbar will map to the 98th percentile of absolute amplitudes. The top 2% of extreme amplitudes (often noise spikes) will be saturated, allowing the subtle reflections in the rest of the data to become highly visible.

### Tool: Amplitude Histogram
If you want to see exactly how your contrast setting affects the data distribution:
1. Navigate to **Tools > View Amplitude Histogram**.
2. A detailed statistical plot will open, displaying the full data distribution (log scale) and a zoomed-in view of the dynamic range actively used by your current contrast setting.
3. This is an excellent QC tool for determining if your data has anomalously high noise spikes ruining your visual dynamic range.

---

## 2. Viewer Controls

* **Flip X-Axis:** Checking the **Flip X** box will instantly reverse the horizontal axis. This is particularly useful if your SEG-Y line was shot right-to-left, but you need to view it left-to-right to match an intersecting line or basemap.
* **Show Grid:** Toggles a light reference grid tied to your X-axis (Trace/Distance) and Y-axis (Time/Depth) ticks.
* **Domain Switcher:** The **Domain** dropdown allows you to label your Y-axis as **Time** (ms) or **Depth** (m). *Note: This only changes the axis label for export and QC purposes; it does not mathematically convert time to depth.*

---

## 3. Rendering & Interpolation Modes

SeisPlotPy offers three distinct ways to draw the seismic pixels to your screen, ranging from raw data rendering to mathematically interpolated high-resolution views.

### 1. Standard Rendering (Default)
When both interpolation boxes are unchecked, the data is rendered exactly as it exists in the array. This is the fastest and truest representation of your data, but zooming in deeply will reveal blocky, pixelated traces.

### 2. Smooth Rendering (Visual Anti-Aliasing)
* **Toggle:** Check the **Smooth** box.
* **What it does:** Enables accelerated Bilinear Interpolation (`SmoothPixmapTransform`). 
* **Best For:** General structural interpretation. It visually smooths out the blocky pixels when zoomed in, making reflectors appear more continuous without altering the underlying data array or slowing down performance.

### 3. High-Res Mode (Mathematical Interpolation)
* **Toggle:** Check the **High Res** box.
* **What it does:** Uses SciPy's cubic spline interpolation engine to mathematically calculate sub-pixel values, expanding the trace array width by 4x before sending it to the graphics card.
* **Best For:** Detailed stratigraphic interpretation of high frequency (sub-bottom) data, subtle fault picking, or preparing data for high-quality PDF/PNG export. There might not be any visual enhancements within standard MCS frequencies.
* ⚠️ **Performance Note:** Because this mode requires heavy mathematical computation, **it is automatically disabled if you attempt to load more than 10 million samples** at once to prevent your computer from freezing. Use this mode when zoomed in on a specific region of interest.
