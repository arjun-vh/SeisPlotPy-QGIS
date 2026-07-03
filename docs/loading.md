# Loading Data

SeisPlotPy handles SEG-Y files dynamically. Rather than loading massive multi-gigabyte files entirely into RAM, it reads headers, determines an optimal preview decimation step, and loads only what is necessary to render the current viewport.

There are two primary ways to load data:

---

## 1. Single SEG-Y Load

This is the standard way to view and interpret a single 2D seismic line.

1. Click **Load Single SEG-Y** (in the sidebar) or **File > Load SEG-Y** (in the menu).
2. Select your `.sgy` or `.segy` file.
3. The plugin uses `segyio` to read the binary and trace headers.
4. It calculates an optimal decimation step to ensure the initial overview loads instantly. 
5. If valid CDP headers are found, the X-axis will automatically switch to CDP.

> **Note on Window Locking:** The **Load Single SEG-Y** button is locked after a file is loaded. SeisPlotPy operates on a "one file per window" philosophy. To load another line while keeping the first open, click the SeisPlotPy icon in the main QGIS toolbar to spawn a new, blank viewer window.

---

## 2. Batch SEG-Y Load

To map an entire survey grid at once without opening a viewer for every single file, use the Batch Loader. See the [Batch Loading](batch_loading.md) page for details.

---

## 3. Fallback / Raw Load Mode

Standard SEG-Y files use a big-endian binary format and IEEE floating-point numbers. However, many legacy or non-standard files deviate from this.

When `segyio` fails to open a file (due to a corrupt binary header, non-standard trace lengths, or severe geometry errors), SeisPlotPy will display a warning:

> *"Unable to open file with segyio. Attempt raw fallback load? (Limited headers available)"*

If you click **Yes**, the plugin completely bypasses `segyio` and uses a custom `numpy.memmap` raw reader with the following automated heuristics:

1. **Endianness Detection:** It reads the format code byte. If the value is absurd (e.g., > 255 in big-endian), it automatically flips the reader to little-endian mode.
2. **IBM Float Conversion:** It first attempts to read the amplitudes as standard IEEE floats. If the maximum absolute amplitude is astronomically high (`> 1e20`) or contains `NaN` values, it automatically assumes the data is in the legacy IBM 32-bit floating-point format and mathematically converts it to IEEE on the fly.
3. **Trace Count Calculation:** Because the binary header might be corrupt, it calculates the total number of traces directly from the file size using the formula: `(file_size - 3600) / (240 + n_samples * 4)`.

### Limitations of Fallback Mode

If your file was loaded using the fallback reader, **all interpretation and processing features still work normally.** The seismic image itself will be completely accurate. 

However, there is a limitation regarding metadata:
* **Fewer Trace Headers:** Only a hardcoded subset of ~30 critical trace headers (like CDP, SourceX, GroupY, offsets, etc.) are extracted. The full 90+ SEG-Y standard header dictionary is not available.
* **No Binary Header Viewer:** The Binary File Header tab in the Header Explorer will be empty.
