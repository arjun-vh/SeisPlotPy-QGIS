# Quick Reference

## Keyboard Shortcuts

The following shortcuts are available when the main SeisPlotPy viewer is active:

| Key | Action | Condition |
|-----|--------|-----------|
| `Esc` | Exit picking mode and return the cursor to standard pan/zoom. | A picking mode is active. |
| `Delete` | Delete the currently selected horizon or fault. | A manager window is open. *(Note: If the interpretation has >5 points, a confirmation dialog appears).* |

---

## Status Bar Messages

The bottom-left label of each SeisPlotPy window displays context-sensitive messages indicating the current state of the plugin:

| Message | Meaning |
|---------|---------|
| `No file loaded` | Initial state. |
| `Loading... Please wait` | Data array is currently being read from disk. |
| `Loaded: name.sgy \| Traces: N` | File was loaded successfully. |
| `Navigation Index Built: N points` | Spatial `cKDTree` is ready for map linking. |
| `Restored: N traces` | The viewer was successfully reopened from a saved QGIS project state. |
| `Horizons auto-saved ✓` | Interpretation successfully saved to sidecar file (disappears after 2s). |
| `Faults auto-saved ✓` | Interpretation successfully saved to sidecar file (disappears after 2s). |
| `Applied AGC` | After AGC processing completes. |
| `Applied Bandpass L-H Hz` | After a bandpass filter completes. |
| `Displayed: Envelope (High Res)` | After running an attribute (append depends on rendering state). |

### Live Coordinate Readout
While hovering your mouse over the seismic plot, the **bottom-right** of the window shows real-time coordinates:
`Header: X.x | Domain: Y.y ms | Amp: Z.z | CRS: E, N | Lat: dd.dddddd, Lon: dd.dddddd`

---

## Sidecar Interpretation Files

Interpretations are not saved inside the SEG-Y file. They are saved in tiny JSON files right next to it in your file system.

| File | Contents | Auto-saved? |
|------|----------|-------------|
| `filename.sgy.horizons.json` | Horizon names, colors, groups, and X/Y point nodes. | **Yes**, on every change. |
| `filename.sgy.faults.json` | Fault names, colors, groups, and X/Y point nodes. | **Yes**, on every change. |

---

## SEG-Y Format Support & Fallback

SeisPlotPy relies on `segyio` for rapid access to standard files, but has a robust fallback raw-reader for non-standard files.

| Case | Behavior |
|------|----------|
| **Standard SEG-Y** (big-endian, IEEE float) | Opened instantly with `segyio`. |
| **Little-endian** | Detected via format code byte > 255; opens in fallback raw mode. |
| **IBM floating-point** | Detected heuristically (if IEEE read results in max amplitude > 1e20 or `NaN`); automatically converted to IEEE in fallback mode. |
| **Corrupt binary header / Unknown geometry** | Initial load fails; fallback mode is offered via dialog. |

**Fallback Mode Limitations:**
Because the raw reader bypasses `segyio` entirely, it only extracts a hardcoded subset of ~30 critical trace headers (like CDP, coordinates, and offsets) instead of the full 90+ field standard. The Header Explorer and QC tools will only show these available fields.

---

## About & Credits

* **Developer:** Arjun V H
* **License:** GPL-3.0
* **DOI:** https://doi.org/10.5281/zenodo.17960131
* **Repository:** https://github.com/arjun-vh/SeisPlotPy-QGIS
