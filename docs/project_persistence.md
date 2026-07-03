# Project Persistence

One of the most powerful features of SeisPlotPy is its deep integration with the QGIS project state. You do not need to manually reload your seismic lines every time you open QGIS.

## Saving Your Work in a QGIS Project

When you save your QGIS project (`.qgz` or `.qgs`), SeisPlotPy automatically intercepts the save event and serializes the complete state of **every open seismic viewer window**.

It writes a custom `<SeisPlotPy>` XML node into your QGIS project file containing:
* The file paths to the SEG-Y files.
* The current visual state: Colormap, Contrast %, X-Axis Reference, Domain (Time/Depth), Flip state, and Grid state.
* The exact viewport extent (X Min/Max, Y Min/Max) and decimation step.

## Reopening a Saved Project

When you reopen that QGIS project later, SeisPlotPy reads the saved state and automatically recreates the QGIS map vectors and re-opens the viewer windows exactly as you left them.

### Dual-Path Resolution Strategy
To make your projects easily shareable with colleagues or portable across different computers, SeisPlotPy saves two paths for every file:
1. **Relative Path:** (e.g., `./data/line1.sgy`). This is calculated relative to where your `.qgz` project file is saved.
2. **Absolute Path:** (e.g., `C:/Projects/Seismic/data/line1.sgy`).

When reopening, the plugin tries to find the file using the relative path first. If the file was moved, it tries the absolute path. If both fail, it will display a file browser dialog asking you to locate the missing SEG-Y file manually.

> **Portability Tip:** The most robust way to organize your work is to keep your `.qgz` project file, all your `.sgy` files, and their sidecar interpretation files in the same parent directory folder. You can then zip that folder and send it to a colleague, and the project will open flawlessly on their machine via the relative paths.

## Double-Click to Re-Open

Sometimes you may close a SeisPlotPy viewer window to clear up screen space while keeping QGIS open.

Because the plugin maintains the spatial link in the background, you can **double-click the seismic line on the QGIS map canvas** at any time. This will instantly re-open the SeisPlotPy viewer for that specific line, restoring all your last-used display settings.

---

## Sidecar Interpretation Files

While visual settings and window states are saved in the QGIS project, your actual scientific interpretations are saved directly next to the SEG-Y file.

* **Horizons:** Saved as `your_filename.sgy.horizons.json`
* **Faults:** Saved as `your_filename.sgy.faults.json`

These files are **auto-saved instantly** every time you add a point, delete a point, or change a color. You never need to hit a "Save" button for interpretations.

> ⚠️ **Warning:** Because these sidecar files travel with the SEG-Y file (not the QGIS project), if you move or rename the `.sgy` file using your operating system, you **must** also move/rename the `.json` files to match, or your interpretations will not load.
