# Batch SEG-Y Loading

If an entire 2D seismic survey grid needs to be analyzed, loading lines one by one into individual viewers is inefficient. The **Batch SEG-Y Load** feature allows loading dozens or hundreds of lines into the QGIS map canvas in a single operation.

## Overview

Batch loading adds multiple 2D seismic lines to the QGIS map canvas **without loading any seismic data into the viewer**. This generates a complete basemap of a survey very quickly. 

After the batch load is complete, the **last selected file** in the batch is automatically linked to the current open viewer window. All the other lines exist purely as spatial map layers in QGIS.

To view the seismic data for any of the other lines, simply **double-click its vector line on the QGIS map**, and a new SeisPlotPy viewer will instantly open for that specific file.

## Step-by-Step Workflow

1. Click **Batch load multiple SEGY** in the left sidebar.
2. A multi-file browser will appear. Select all the `.sgy` or `.segy` files belonging to the survey grid.
3. **Header Uniformity Notice:** You will see a warning reminding you that all selected files must share the same coordinate header byte positions and the same Coordinate Reference System (CRS). Click **Yes** to continue.
4. The plugin will read the headers from the *first* file in your selection and present the **Geometry Dialog**.
5. Select the **X/Y coordinate headers** (e.g., `CDP_X` and `CDP_Y`) and the scaling method.
6. Select the **Coordinate Reference System (CRS)** using the standard QGIS projection selector.
7. A progress dialog will appear, looping through each file. It parses the headers, computes the spatial tree, and generates a QGIS vector line for each file. 
    * *You can click **Abort** at any time to stop the process. Files processed up to that point will remain on the map.*
8. Once complete, a summary message will report how many lines were successfully loaded to the map canvas.

## Limitations & Requirements

* **Uniform Header Layout:** All files in the batch *must* store their coordinates in the exact same byte locations. If Line A has X coordinates at byte 73, but Line B has X coordinates at byte 181, Line B will be mapped to the wrong location (or fail entirely).
* **Uniform CRS:** All files must belong to the same CRS.
* **Distance Axis Calculation:** To save processing time during the batch loop, the cumulative distance axis is only auto-calculated for the *last linked file*. However, because the plugin saves your geometry parameters to the QGIS layer, **when you double-click any other batch layer to open it, SeisPlotPy automatically recalculates and populates its cumulative distance axis on the fly.** You do not need to set up the geometry again.

---

> **Acknowledgment:** The Batch SEG-Y Loading feature was developed from an initial prototype script contributed by **Mr. Muhammed Anshif K K**.
