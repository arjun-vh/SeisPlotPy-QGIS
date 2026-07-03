# Exporting

SeisPlotPy provides tools to export your seismic data either as high-quality visual figures for publication, or as mathematical data subsets for use in other geophysical software.

---

## 1. Exporting Publication Figures

When you need to include a seismic section in a report, presentation, or academic paper, you can generate high-resolution image exports directly from the SeisPlotPy viewer. 

The export engine uses Matplotlib in the background to ensure publication-quality vector and raster graphics, entirely independent of your screen resolution.

### How to Export a Figure
1. In the left sidebar, locate the **Export** group. (Alternatively, use `File > Export PDF/PNG`).
2. **Set the Dimensions:** Enter your exact desired physical size in the **W (in)** and **H (in)** spinboxes (e.g., 8 x 6 inches).
3. **Match Aspect Ratio:** Click this button to automatically resize the SeisPlotPy window so the on-screen pixel ratio exactly matches your target physical dimensions. This ensures that what you see on screen is exactly what gets exported.
4. Click **Export Figure**.
5. Choose your format: **PDF** (default), **SVG**, or **PNG**.

### Rendering Quality Features
* **Vector Text Editing:** If you export to PDF or SVG, the plugin uses TrueType embedding (`pdf.fonttype = 42`, `svg.fonttype = 'none'`). This means all axis labels and tick numbers are fully editable text objects in Adobe Illustrator or Inkscape (they are not rasterized or converted to paths).
* **Lanczos Interpolation:** The seismic image itself is rendered using high-quality Lanczos interpolation. This provides a much smoother, mathematically superior image compared to the standard viewer's bilinear smoothing.
* **Horizon and Fault Rendering:** To distinguish interpretations on static exports, horizons are drawn as plain lines, while faults are drawn with subtle dot markers at each picked node (`linestyle='-', marker='.', markersize=2`).

---

## 2. Exporting a SEG-Y Subset

If you have a massive multi-gigabyte SEG-Y file but only want to share a specific 500-trace structural feature with a colleague, you can export a raw SEG-Y subset. 

This creates a mathematically perfect, truncated copy of your data without re-sampling or altering the amplitudes.

### How to Export a Subset
1. Use the viewport controls (or pan/zoom) to frame the exact extent of traces and time/depth you want to export.
2. Navigate to **File > Export SEG-Y Subset...**.
3. A confirmation dialog will appear, stating the number of traces and the time/depth range that will be written.
4. Click **Yes** and select a save destination.

### Trace Sequence Reset
When the subset is written to the new file, SeisPlotPy intelligently resets the `TraceSequenceFile` trace header (bytes 5–8) to consecutive integers starting at 1. This ensures that external software (like OpendTect or Petrel) correctly identifies the new file as a continuous, unbroken sequence of traces, rather than a fragmented subset.
