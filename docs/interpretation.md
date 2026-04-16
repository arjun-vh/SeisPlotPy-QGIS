# Horizon & Fault Interpretation

SeisPlotPy provides robust, systematic tools for digitizing seismic horizons and faults directly on the interactive plot. These interpretations can be dynamically manipulated, exported for external analysis, or published back to your QGIS map canvas as spatial vector layers.

## 1. The Interpretation Managers

To begin interpreting, open either the **Horizon Manager** or the **Fault Manager** from the `Tools` menu. 

Both managers function similarly and act as your control center for interpretations. From this interface, you can:
* **Add New:** Create a new horizon/fault and assign it a distinct name and color.
* **Toggle Visibility:** Show or hide specific interpretations using the checkbox.
* **Change Color:** Click the color box to update the visual styling on the fly.
* **Delete:** Remove an interpretation completely. *(Note: Deleting an interpretation with more than 5 points will prompt a confirmation dialog to prevent accidental data loss).*

---

## 2. Picking and Editing

Once you have created a horizon or fault in the manager, you can begin digitizing it on the seismic viewer.

### How to Pick
1. Select the target horizon/fault in the Manager list.
2. Click the **Pick** button (or toggle the picking state). The seismic viewer cursor will change to a crosshair.
3. **Left-Click** on the seismic image to drop a node. 

### Editing & Removing Points
If you make a mistake while picking, you can easily correct it without starting over:
* **For Horizons:** **Right-Click** near a node to delete the *closest point*.
* **For Faults:** **Right-Click** anywhere on the plot to *undo the last point* you added.

### Keyboard Shortcuts
For an efficient workflow, utilize these keyboard shortcuts:
* Press `Esc` to instantly exit picking mode and return to standard viewer navigation.
* Press `Delete` to remove the currently selected horizon or fault.

> **Auto-Saving:** SeisPlotPy automatically saves your interpretations to sidecar `.json` files (e.g., `your_file.sgy.horizons.json` and `your_file.sgy.faults.json`) whenever you add points, change colors, or toggle visibility. You will see a subtle confirmation message in the status bar upon a successful save.

---

## 3. Horizon Flattening (Dynamic Warping)

One of SeisPlotPy's advanced analytical features is the ability to structurally flatten the seismic data to a specific reference horizon, aiding in stratigraphic analysis.

1. In the Horizon Manager, check the **Flatten** box next to your target reference horizon.
2. The plugin will calculate the mean depth/time of your horizon and dynamically shift every trace to flatten that specific reflector.
3. All other visible horizons and faults will automatically warp to match the newly flattened coordinate space.

> ⚠️ **Important Safety Mechanism:** While "Flattened" mode is active, **all interpretation picking is disabled**. This is a deliberate safeguard to prevent spatial coordinates from becoming corrupted while the visual space is warped. To resume picking, simply uncheck the Flatten box to return to the native structural view.

---

## 4. Exporting Interpretations to CSV

You can extract your interpretations to standard `.csv` files for use in external modeling software or Python scripts. The export engine safely handles edge cases to ensure your data perfectly matches the file boundaries.

### Export Options
* **Single Export:** Select an interpretation in the Manager and click the **Export** button.
* **Batch Export:** Click **Export All** to automatically save all *visible* interpretations to a selected directory. The system will auto-number files if multiple horizons share the same name.

### Header Integration
When you initiate an export, SeisPlotPy will prompt you with a Header Selection Dialog. This allows you to append raw SEG-Y trace headers (e.g., `CDP_X`, `CDP_Y`, `Elevation`) directly to your interpretation points. 

The resulting CSV will include:
1.  **Trace Index:** The exact sequential trace number.
2.  **Current Domain:** The Y-axis value (Time in ms, or Depth in m).
3.  **Mapped X-Coordinate:** If you are viewing the data by CDP or Cumulative Distance, those real-world spatial values will be included.
4.  **Selected Headers:** Any additional SEG-Y headers you selected during the export prompt.

---

## 5. Publishing to the Map Canvas

Because SeisPlotPy is integrated with QGIS, your interpretations aren't trapped in the seismic viewer. 

By clicking the **Map** button in either manager, SeisPlotPy will immediately generate a new temporary vector layer (`LineString`) in your QGIS project containing your interpretation. 

* The plugin automatically extracts the geographic coordinates (utilizing your established `CDP_X`/`CDP_Y` mapping and scalars) to plot the polyline in its exact real-world location.
* The layer includes standard attribute data, such as the horizon name, average time/depth, and the bounding trace indices.
