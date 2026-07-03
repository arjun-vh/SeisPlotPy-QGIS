# Display Controls & High-Res Rendering

SeisPlotPy is built on `pyqtgraph` to ensure extremely fast rendering of large 2D arrays. To help the user highlight specific structural or stratigraphic features, the plugin offers a comprehensive suite of real-time display controls.

All display settings are located in the left-hand sidebar of the main SeisPlotPy window.

---

## 1. The Active Viewport Controls

Because a single seismic line can contain tens of thousands of traces, the top section of the sidebar allows the user to define exactly what subset of data is fetched from disk and rendered.

* **X-Axis Reference:** This dropdown dictates the horizontal coordinate space. It starts at "Trace Index" upon initial load. As the user uses other features, new options appear dynamically (e.g., CDP, other headers, or "Cumulative Distance" after setting up geometry).
* **X-Axis Range and Y-Axis Range:** These min/max spinboxes define the precise bounding box of data to load. The user can edit the numbers manually and click **Apply / Reload** to zoom directly to an exact area of interest.
* **Decimation (Step):** By default, the plugin uses a smart auto-step (`trace_count / 2000`) to load massive files instantly, but steps down to 1 (full resolution) when zooming in closely. To override this, check the **Manual** checkbox and set a specific step value.
* **Reset View:** Restores the viewport to encompass the entire file extent.

---

## 2. Color & Contrast Management

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
To see exactly how the contrast setting affects the data distribution, navigate to **Tools > View Amplitude Histogram**. A dual-panel Matplotlib figure will open:
   * **Top panel:** The full amplitude distribution on a log scale, with red vertical lines showing the clip thresholds and shaded regions showing the saturated data.
   * **Bottom panel:** A zoomed-in view of the dynamic range actively used by your current contrast setting.
   * **Statistics Textbox:** Displays Min, Max, Mean, Std, the absolute clip threshold value, and the percentage of data saturated.
   * The user can save this QC plot to PNG or PDF using the toolbar buttons.

### Color Legend
Checking the **Show Color Legend** box toggles an interactive color bar to the right of the plot. Note that the color bar is read-only; the active colormap must be changed via the Colormap dropdown.

---

## 3. General Viewer Controls

* **Flip X-Axis:** Checking the **Flip X** box will instantly reverse the horizontal axis. This is particularly useful if the SEG-Y line was shot right-to-left, but you need to view it left-to-right to match an intersecting line or basemap.
* **Show Grid:** Toggles a light reference grid tied to your X-axis and Y-axis ticks. Click the **⚙ (Gear)** button next to it to open the Grid Settings, where you can independently toggle X/Y lines, adjust their transparency (alpha).
* **Domain Switcher:** The **Domain** dropdown allows you to label your Y-axis as **Time** (ms) or **Depth** (m). *Note: This changes the axis label for export and QC purposes in both the viewer and exported figures; it does not mathematically convert time to depth.*

---

## 4. Rendering & Interpolation Modes

SeisPlotPy offers three distinct ways to draw the seismic pixels to your screen.

### 1. Standard Rendering (Default)
When both interpolation boxes are unchecked, the data is rendered exactly as it exists in the array. This is the fastest and truest representation of your data, but zooming in deeply will reveal blocky, pixelated traces.

### 2. Smooth Rendering (Visual Anti-Aliasing)
* **Toggle:** Check the **Smooth** box.
* **What it does:** Enables accelerated hardware Bilinear Interpolation. 
* **Best For:** General structural interpretation. It visually smooths out the blocky pixels when zoomed in without altering the underlying data array.

### 3. High-Res Mode (Mathematical Interpolation)
* **Toggle:** Check the **High Res** box.
* **What it does:** Uses SciPy's cubic spline interpolation engine to mathematically calculate sub-pixel values before sending the data to the graphics card.
* **Configuration:** Click the **⚙ (Gear)** button to open the High-Res Configuration dialog. Here you can set independent **Vertical** and **Horizontal** multipliers (1x, 2x, 4x, or 8x).
* **Safety Limits:** Because spline interpolation requires heavy RAM computation:
   1. It is automatically disabled if you attempt to load more data than the configurable Safety Limit (default 10 million samples).
   2. It is automatically disabled if your current decimation step is > 1. You cannot mathematically interpolate decimated overview data.

---

## 5. Live Coordinate Status Bar

As you move your mouse across the seismic image,the values in the bottom-right corner of the window will update in real-time. It provides a real-time readout of exact data parameters under the cursor:

`Header: 1250 | Domain: 2450.5 ms | Amp: -4.32e+03 | CRS: 543021, 8765432 | Lat: 72.1234, Lon: -14.5678`

* **Header:** The X-axis value (Trace, CDP, or Distance).
* **Domain:** The precise Y-axis value (Time or Depth).
* **Amp:** The raw seismic amplitude pulled directly from the in-memory array.
* **CRS:** The Easting/Northing projected map coordinate (available after setting up geometry).
* **Lat/Lon:** The geographic coordinate dynamically projected to WGS84 on the fly.
