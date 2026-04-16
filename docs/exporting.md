# Exporting Data & Figures

Whether you need a high-resolution, publication-ready figure or a cropped subset of your raw SEG-Y data, the export engine handles it safely and accurately.

---

## 1. Exporting Publication-Ready Figures

If you have configured the perfect view of a structural feature or attribute, you can export the plot directly to an image or vector file.

1. Adjust your view to the desired bounds using the navigation spinboxes or by panning/zooming.
2. Navigate to **File > Export PDF/PNG**.
3. **Set Resolution:** A dialog will prompt you to enter the desired DPI (Dots Per Inch). The default is `600` DPI, which is standard for academic and industry publications.
4. **Choose Format:** You can save the file as a `PNG` (raster image), `PDF`, or `SVG` (vector graphics).

### What gets exported?
The export engine is WYSIWYG (What You See Is What You Get) but at much higher fidelity. It automatically captures:
* **Visual Settings:** Your current colormap, percentile contrast clipping, X-axis flipping, and grid visibility are all honored.
* **Axes Labels:** The X-axis will accurately reflect your current domain (Trace Index, CDP, or Cumulative Distance).
* **Interpretations:** All currently visible horizons and faults are drawn over the seismic data. The export engine automatically maps the interpretation coordinates to match your current X-axis domain so they align perfectly.

> 🖌️ **Vector Editing Support:** If you export to PDF or SVG, SeisPlotPy explicitly preserves the text elements (using standard font typing) and vector lines. This allows you to open the exported figure in Adobe Illustrator or Inkscape to fine-tune labels, line weights, and layout without any pixelation.

---

## 2. Exporting SEG-Y Subsets

Massive 2D seismic lines can be cumbersome to share or process. SeisPlotPy allows you to crop the data spatially and save a new, fully compliant SEG-Y file containing only your Region of Interest (ROI).

1. Identify the starting and ending trace numbers you wish to keep.
2. Navigate to **File > Export SEG-Y Subset...**
3. A dialog will prompt you for the **Start Trace** and **End Trace**, as well as the output file location.
4. Click **OK**.

### How the Subset Engine Works
To ensure the new file is safe and readable by any other seismic software, SeisPlotPy performs a non-destructive extraction:
* It streams the requested traces from the original file, preventing memory overloads on massive datasets.
* It copies the original binary header, safely updating bytes 3213-3214 to reflect the new total trace count.
* It automatically enforces Big-Endian (`>`) byte order for maximum compatibility.
* *Optional:* If you previously edited the text header (via **Tools > Header Utilities > View Text Header**), your custom EBCDIC text will be embedded into the new subset file.

---

## 3. Exporting Tabular Data (CSV)

SeisPlotPy offers robust CSV exporting for moving numerical data into Python, Excel, or modeling platforms. 

* **Exporting Interpretations (Horizons & Faults):** You can extract the precise X/Y coordinates of your interpretations, appended with any raw SEG-Y trace headers you choose. *(See the [Interpretation Guide](interpretation.md) for details).*
* **Exporting Trace Headers:** You can dump the entire metadata table (or selected columns) of your SEG-Y file. *(See the [Header Utilities Guide](headers.md) for details).*
