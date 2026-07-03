# Header Utilities

The SEG-Y format relies heavily on binary and trace headers to store critical metadata such as coordinate geometry, scaling factors, shot records, and sample rates. SeisPlotPy provides comprehensive tools to inspect, visualize, modify, and export this header data.

---

## 1. Header Explorer

To inspect the raw header values encoded in your file, navigate to **Tools > Header Utilities > Header Explorer (Binary/Trace)**.

The Explorer window contains two tabs:

### Trace Headers Tab
This is a spreadsheet-like view of every trace in the file. 
* **Rows:** Represent individual seismic traces.
* **Columns:** Represent the individual trace header fields (e.g., `CDP_X`, `Elevation`, `SourceGroupScalar`).
* Because a file may contain 100,000+ traces, the Explorer pages the data. Use the scrollbar to fetch the next block of traces efficiently.

### Binary File Header Tab
The SEG-Y 400-byte binary header contains global survey parameters. This tab displays a decoded, human-readable list of over 30 critical fields, including:
`Job_ID`, `LineNumber`, `DataTracePerRecord`, `SampleInterval`, `DataSampleFormatCode`, `MeasurementSystem`, and `SEGYRevisionNumber`.

> **Fallback Mode Limitation:** If your file was opened using the raw fallback reader, the Binary File Header tab will be unavailable. Furthermore, the Trace Headers tab will only display a hardcoded subset of ~30 core header fields, rather than the full 90+ field standard.

---

## 2. Trace Header QC Plot

Graphing the headers makes it immediately obvious if you have bad coordinates, missing scalars, or corrupted trace headers.

1. Navigate to **Tools > Header Utilities > Trace Header QC Plot**.
2. Select a header field from the dropdown.
3. SeisPlotPy instantly extracts that byte location from every trace and graphs the value sequentially (Trace Index vs Header Value).
4. You can use the standard Matplotlib controls to zoom, pan, and save the QC graph.

---

## 3. View Text Header

The SEG-Y EBCDIC (or ASCII) text header contains 3200 bytes of free-form text, usually containing the observer's log, processing history, and CRS definitions.

### Editing the Text Header
SeisPlotPy allows you to view and directly edit this text header inside a modeless dialog (which means you can leave it open while interacting with the main window).

1. Navigate to **Tools > Header Utilities > View Text Header**.
2. You can type directly into the text box to update the observer log or correct a typo.
3. Click **Save As New File...** to write your changes. 
4. The plugin will automatically sanitize your text (enforcing the strict 40 lines × 75 characters format, replacing tabs with spaces, and stripping old `C xx` prefixes) before automatically opening the [SEG-Y Subset Export](exporting.md) dialog. Your modified text header will be embedded into the newly exported file.

---

## 4. Bulk Header Modification

If you need to fix coordinates or apply a static shift across an entire 2D line, you can export the headers, manipulate them in Excel or Python, and patch them back into a new SEG-Y file.

### Step 1: Export Headers to CSV
1. Navigate to **Tools > Header Utilities > Export Headers to CSV...**
2. Select the specific header fields you want to extract.
3. Click Export. SeisPlotPy generates a CSV with one row per trace.

### Step 2: Edit the CSV
Open the CSV in your preferred spreadsheet software. You can perform bulk calculations (e.g., converting feet to meters, adding a constant static shift, or interpolating missing X/Y coordinates). Save the file, ensuring you don't change the column names.

### Step 3: Patch Headers from CSV
1. Navigate to **Tools > Header Utilities > Patch Headers from CSV...**
2. Select your modified CSV file.
3. SeisPlotPy will read the CSV, map the column names back to standard SEG-Y byte locations, and generate a **new SEG-Y file**. 
4. *Note: SeisPlotPy never overwrites your original data. A new file is always created during a patching operation.*
