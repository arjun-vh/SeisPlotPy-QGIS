# Horizon & Fault Interpretation

SeisPlotPy provides systematic tools for digitizing seismic horizons and faults directly on the interactive plot. These interpretations can be dynamically manipulated, exported for external analysis, or published back to QGIS map canvas as spatial vector layers.

## 1. The Interpretation Managers

To begin interpreting, open either the **Horizon Manager** or the **Fault Manager** from the `Tools` menu. 

Both managers feature a data table with the following columns: Active (radio button), Visibility (checkbox), **Name**, **Group**, Color, Point count, **Flat** (Horizons only), and Actions.

* **Add New:** Create a new interpretation and assign it a distinct name.
* **Group:** You can categorize interpretations (e.g., "Seafloor", "Target"). During batch export, filenames are structured as `{Group}_{Name}.csv`.
* **Toggle Visibility:** Show or hide specific interpretations using the checkbox.
* **Change Color:** Click the color box to update the visual styling on the fly.
* **Delete:** Remove an interpretation completely. *(Note: Deleting an interpretation with >5 points prompts a confirmation dialog).*
* **Import CSV:** Load pre-picked interpretations from an external file. The expected format is two columns (Trace Index, Time/Depth) with no specific header row (the first row is skipped).

---

## 2. Picking and Editing

Once you have created an entry in the manager, you can begin digitizing.

### How to Pick
1. Select the target horizon/fault in the Manager list.
2. Click the **Pick** button (a toggle button that changes color when active). The seismic viewer cursor will change to a crosshair. 
   *(Note: Activating picking for a horizon automatically deactivates fault picking, and vice versa).*
3. **Left-Click** on the seismic image to drop a node. 

### Editing & Removing Points
If you make a mistake while picking, you can easily correct it without starting over:
* **For Horizons:** **Right-Click** near a node to delete the *closest point*.
* **For Faults:** **Right-Click** anywhere on the plot to *undo the last point* you added.

### Keyboard Shortcuts
* Press `Esc` to instantly exit picking mode and return to standard viewer navigation.
* Press `Delete` to remove the currently selected horizon or fault.

> **Auto-Saving:** SeisPlotPy automatically saves your interpretations to sidecar `.json` files (e.g., `your_file.sgy.horizons.json` and `your_file.sgy.faults.json`) whenever you add points, change colors, or toggle visibility. You never need to hit a manual save button.

---

## 3. Horizon Flattening (Dynamic Warping)

One of SeisPlotPy's advanced analytical features is the ability to structurally flatten the seismic data to a specific reference horizon, aiding in stratigraphic analysis.

1. In the Horizon Manager, check the **Flat** box next to your target reference horizon.
2. The plugin will calculate the mean depth/time of your horizon and dynamically shift every trace to flatten that specific reflector.
3. All other visible horizons and faults will automatically warp to match the newly flattened coordinate space.

> ⚠️ **Important Safety Mechanism:** While "Flattened" mode is active, **all interpretation picking (both Horizons and Faults) is completely disabled**. A warning dialog will alert you if you attempt to pick. This is a deliberate safeguard to prevent spatial coordinates from becoming corrupted while the visual space is warped. Uncheck the Flat box to return to the native structural view and resume picking.

---

## 4. Exporting Interpretations to CSV

You can extract your interpretations to standard `.csv` files for use in external modeling software or Python scripts. 

### Horizon Exports vs Fault Exports
There is a fundamental difference in how horizons and faults are exported:
* **Horizons:** Horizons are continuous reflectors. When exported, they are **interpolated at every trace** from `min_idx` to `max_idx`. Furthermore, the export includes an **Amplitude** column—the raw seismic amplitude extracted from the data array at every single point, perfect for AVO analysis.
* **Faults:** Faults are discrete cuts. When exported, they output *only the exact nodes you picked*, in the exact order you picked them. No interpolation or amplitude extraction occurs.

### Batch Export Options
Click **Export All** to automatically save all *visible* interpretations to a directory. The system will auto-number files if multiple horizons share the same name.

### Header Integration
When you initiate an export, you can select SEG-Y trace headers (e.g., `CDP_X`, `CDP_Y`, `Elevation`) to append directly to your interpretation points.

---

## 5. Publishing to the Map Canvas

Because SeisPlotPy is integrated with QGIS, your interpretations aren't trapped in the seismic viewer. By clicking the **Map** button in either manager, SeisPlotPy immediately generates a new vector layer in your QGIS project using the established `CDP_X`/`CDP_Y` geometry.

### Horizons → Point Layer
A published horizon creates a **Point** vector layer containing one point per interpolated trace.
* **Attributes:** `Trace`, `Time_Depth`, `Amplitude`, **plus all available SEG-Y trace headers**.
* This is a powerful GIS output: you can use its attributes for spatial interpolation (kriging, IDW) or color by amplitude in a seismic-attribute map directly inside QGIS.

### Faults → LineString Layer
A published fault creates a single **LineString** layer (polyline).
* **Attributes:** `FaultName`, `Points` (count).
