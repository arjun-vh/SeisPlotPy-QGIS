# Header Exploration & Utilities

SEG-Y files rely entirely on embedded metadata (headers) to establish coordinate geometry, sampling rates, and trace numbering. Incorrect or missing headers are the most common cause of spatial and processing errors. 

SeisPlotPy provides a comprehensive suite of tools to inspect, visualize, and fix these headers directly within the plugin. All tools are located under **Tools > Header Utilities**.

---

## 1. Visualizing & QC'ing Headers

Before attempting to fix or map your data, it is crucial to understand what metadata is actually stored in your file.

### View Text Header (EBCDIC / ASCII)
The textual file header (usually 3200 bytes) contains the human-readable survey history, acquisition parameters, and processing notes.
1. Navigate to **Tools > Header Utilities > View Text Header**.
2. A window will display the parsed text. 
3. **Editing:** If you notice a typo or want to append your own processing notes, you can type directly into this window. Click **Save As New File...** to export a copy of the SEG-Y with your updated text header.

### Header Explorer
If you need to inspect the raw binary or trace headers in a structured format:
1. Navigate to **Tools > Header Utilities > Header Explorer (Binary/Trace)**.
2. This opens a tabular spreadsheet view of your file's metadata, allowing you to quickly scroll through trace indices and see the exact integer or float values stored in every byte location.

### Trace Header QC Plot
Ggraphing the headers makes it  obvious if you have bad coordinates, dropped traces, or static shift errors.
1. Navigate to **Tools > Header Utilities > Trace Header QC Plot**.
2. Select a header (e.g., `CDP_X` or `Elevation`) from the dropdown.
3. SeisPlotPy will plot the header values on the Y-axis against the Trace Index on the X-axis. Look for sudden spikes, gaps, or zeros that indicate corrupted metadata.

---

## 2. Exporting Headers to CSV

If you need to analyze your geometry in external software (like Python, Excel, or standalone QGIS), you can extract the trace headers to a standard CSV file.

1. Navigate to **Tools > Header Utilities > Export Headers to CSV...**.
2. A dialog will appear listing all available headers in your SEG-Y file.
3. Check the boxes next to the headers you want to extract (e.g., `TraceNumber`, `CDP_X`, `CDP_Y`).
4. Click OK and choose a save location. The resulting CSV will contain one row per trace.

---

## 3. Patching Headers (Advanced)

If your SEG-Y file is missing spatial coordinates, or the coordinates were written in the wrong byte locations, SeisPlotPy allows you to "patch" the file using an external CSV. 

*Common Use Case:* You have a raw SEG-Y file and a separate CSV navigation file containing the shotpoint coordinates.

1. Navigate to **Tools > Header Utilities > Patch Headers from CSV...**.
2. **Select Inputs:** Provide the path to your source CSV and the path for the new target SEG-Y file.
3. **Configure the Mapping:** You must map the columns in your CSV to the specific SEG-Y byte locations (e.g., mapping your CSV's `Easting` column trace header `SourceX`).
4. **Execute:** Click OK to start the patching process.

> 🛡️ **Safety Mechanism (Non-Destructive Patching):** > To protect your original data, SeisPlotPy **never** overwrites your source SEG-Y file. The patching engine streams the trace data from your original file, swaps the specified header bytes, automatically standardizes the Endianness (to Big-Endian `>`), and writes an entirely new, structurally sound SEG-Y file to your hard drive.
