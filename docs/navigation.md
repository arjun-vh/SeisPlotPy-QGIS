# Navigation & Spatial Linking

SeisPlotPy is designed to bridge the gap between raw geophysical arrays (traces and amplitudes) and geographic positioning. By extracting coordinate data from your SEG-Y trace headers, the plugin generates a spatial index, calculates true cumulative distances, and seamlessly links the seismic viewer to the QGIS map canvas.

---

## 1. Setting Up Geometry & Distance

When you first load a SEG-Y file, the X-axis defaults to the sequential **Trace Index**. To view your data in geographic space, you must configure the geometry.

1. Navigate to **Tools > Setup Geometry / Distance**.
2. A dialog will appear asking you to map your coordinate headers.
   * **X-Coordinate Header:** Typically `CDP_X`, `SourceX`, or `GroupX`.
   * **Y-Coordinate Header:** Typically `CDP_Y`, `SourceY`, or `GroupY`.
3. **Apply Scalar:** SEG-Y coordinates are often stored as integers multiplied by a scaling factor. 
   * **Use Header:** Check this to automatically apply the scalar found in the file. The dropdown below it lets you select the specific scalar header key (usually `SourceGroupScalar`).
   * **Manual Value:** If your file lacks a scalar header, uncheck the box and enter the known multiplier manually (e.g., `0.1` or `1.0`).
4. Click **OK** to proceed to CRS Selection.

---

## 2. Coordinate Reference Systems (CRS)

Immediately after setting the geometry, QGIS will prompt you to select a Coordinate Reference System (CRS) for your data.

It is critical that you select the correct CRS that matches how the data was recorded:
* **Projected CRS (Meters/Feet):** E.g., UTM Zones. Use this if your coordinates are in standard X/Y grids.
* **Geographic CRS (Degrees):** E.g., WGS 84. Use this if your coordinates are in Longitude/Latitude.

> ⚠️ **Warning:** If SeisPlotPy detects that your SEG-Y binary header indicates `CoordinateUnits = 2` (Seconds of Arc), it will automatically convert the raw coordinates to decimal degrees. If you select a Projected CRS in this scenario, the plugin will abort the operation with a mismatch warning.

---

## 3. The Map Canvas Vector Layer

Once the geometry is successfully calculated, SeisPlotPy performs three background actions:

1. **Calculates Cumulative Distance:** A highly accurate line-length array is generated. It uses the Haversine formula (if your CRS is geographic) or standard Euclidean math (if projected). Your X-axis will automatically switch to **Cumulative Distance**. If the total length exceeds 10,000 meters, it will auto-scale the axis labels to **km**.
2. **Builds a Spatial Tree:** A high-speed navigation index (`cKDTree`) is built on a decimated coordinate array to map screen pixels to real-world coordinates instantly.
3. **Generates a QGIS Map Layer:** A new temporary vector polyline representing your 2D seismic survey is added to your QGIS map canvas.

---

## 4. Interactive Navigation

With the spatial link established, you can interact continuously between the QGIS main window and the SeisPlotPy viewer.

### Activating the Link/Nav Tool
> ⚠️ **Important:** Before map-linked navigation works, you must activate the **Link/Nav Tool**. Click the crosshair/cursor icon in the QGIS toolbar (it becomes highlighted when active). Without this, hovering over the map will not update the SeisPlotPy viewer.

### The QGIS Map Tool
When the Link/Nav Tool is active on your QGIS canvas:
* **Hover:** As you move your mouse along the seismic line on the map, SeisPlotPy calculates the nearest trace. **Dynamic Tolerance** is used to ensure accuracy: a 10-pixel screen offset is transformed into your layer's specific CRS to calculate snap distance. Look at the status bar at the bottom of the SeisPlotPy window—it will dynamically update to show exactly what trace you are hovering over on the map.
* **Double-Click:** If your SeisPlotPy window is closed, double-clicking anywhere near the seismic line on the QGIS map will instantly open the viewer again to the front of your screen.

### The Extent Highlighter (Red Line)
As you zoom and pan horizontally within the SeisPlotPy viewer, a **thick red highlight line** will dynamically appear on top of the vector layer in QGIS. 

This visualizes exactly which segment of the full 2D line you are currently looking at on your screen, ensuring you never lose your geographic context while investigating small-scale structural features.
