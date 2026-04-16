# Navigation & Spatial Linking

SeisPlotPy is designed to bridge the gap between raw geophysical arrays (traces and amplitudes) and geographic positioning. By extracting coordinate data from your SEG-Y trace headers, the plugin generates a spatial index, calculates true cumulative distances, and seamlessly links your seismic viewer to the QGIS map canvas.

---

## 1. Setting Up Geometry & Distance

When you first load a SEG-Y file, the X-axis defaults to the sequential **Trace Index**. To view your data in geographic space, you must configure the geometry.

1. Navigate to **Tools > Setup Geometry / Distance**.
2. A dialog will appear asking you to map your coordinate headers.
   * **X-Coordinate Header:** Typically `CDP_X`, `SourceX`, or `GroupX`.
   * **Y-Coordinate Header:** Typically `CDP_Y`, `SourceY`, or `GroupY`.
3. **Apply Scalar:** SEG-Y coordinates are often stored as integers multiplied by a scaling factor. 
   * **Use Header:** Check this to automatically apply the scalar found in the file (usually `Source_Group_Scalar`).
   * **Manual Value:** If your file lacks a scalar header, uncheck the box and enter the known multiplier manually (e.g., `0.1` or `1.0`).
4. Click **OK** to proceed to CRS Selection.

---

## 2. Coordinate Reference Systems (CRS)

Immediately after setting the geometry, QGIS will prompt you to select a Coordinate Reference System (CRS) for your data.

It is critical that you select the correct CRS that matches how the data was recorded:
* **Projected CRS (Meters/Feet):** E.g., UTM Zones. Use this if your coordinates are in standard X/Y grids.
* **Geographic CRS (Degrees/Arc-Seconds):** E.g., WGS 84. Use this if your coordinates are in Longitude/Latitude.

> ⚠️ **Warning:** If SeisPlotPy detects that your SEG-Y coordinate units are marked as Arc-Seconds (Unit 2 in the binary header), but you select a Projected CRS, it will abort the calculation and warn you. 
---

## 3. The Map Canvas Vector Layer

Once the geometry is successfully calculated, SeisPlotPy performs three actions:

1. **Calculates Cumulative Distance:** A highly accurate line-length array is generated (using the Haversine formula for geographic coordinates, or Euclidean math for projected coordinates). Your X-axis will automatically switch to **Cumulative Distance** (in meters or kilometers).
2. **Builds a Spatial Tree:** A high-speed navigation index (`cKDTree`) is built in the background to map screen pixels to real-world coordinates.
3. **Generates a QGIS Map Layer:** A new vector layer (a polyline representing your 2D seismic survey) is added to your QGIS map canvas.

---

## 4. Interactive Navigation

With the spatial link established, you can interact continuously between the QGIS main window and the SeisPlotPy viewer.

### The QGIS Map Tool
When SeisPlotPy is running, an active map tool is engaged on your QGIS canvas (indicated by a crosshair cursor).
* **Hover:** As you move your mouse along the seismic line on the QGIS map, SeisPlotPy calculates the nearest trace. Look at the status bar at the bottom of the SeisPlotPy window—it will dynamically update to show the exact Map Coordinates and corresponding Seismic Trace under your mouse.
* **Double-Click:** If your SeisPlotPy window is closed, double-clicking anywhere near the seismic line on the QGIS map will instantly open the viewer again to the front of your screen.

### The Extent Highlighter (Red Line)
As you zoom and pan horizontally within the SeisPlotPy viewer, a **thick red highlight line** will dynamically appear on top of the vector layer in QGIS. 

This visualizes exactly which segment of the full 2D line you are currently looking at on your screen, ensuring you never lose your geographic context while investigating small-scale structural features.
