# Seismic Processing & Attributes

SeisPlotPy includes a suite of high-performance signal processing and complex trace attribute calculations that can be applied to your data on the fly. These are excellent for highlighting specific stratigraphic features, suppressing noise, or identifying bright spots.

> **A Note on Raw / Fallback Mode:** If your SEG-Y was opened using the raw fallback reader (because `segyio` failed), all processing and attributes still work perfectly. The seismic data array itself is mathematically identical.

---

## 1. Core Processing Tools

### Apply AGC (Automatic Gain Control)
Seismic data often suffers from spherical divergence (energy decays over time/depth). AGC calculates a sliding RMS window and scales the amplitude trace by trace to normalize energy distribution, making deep, low-amplitude reflectors visible.

* **Usage:** `Processing > Apply AGC`. 
* **Window Size:** A prompt will ask for a Window Size in milliseconds. The range is **10 to 5000 ms**, with a default of 500 ms. Smaller windows balance energy aggressively (often washing out true relative amplitudes); larger windows preserve more of the relative amplitude character.

### Bandpass Filter
A standard zero-phase 4th-order Butterworth filter (`scipy.signal.butter` + `filtfilt`) to remove low-frequency swell noise or high-frequency acquisition noise.

* **Usage:** `Processing > Bandpass Filter`.
* **Nyquist Safety:** The dialog automatically computes the Nyquist frequency from your file's sample interval. It enforces a maximum High Cut limit at **90% of Nyquist**. You cannot enter a frequency above this cap.
* **Defaults:** Low Cut defaults to 8 Hz; High Cut defaults to 60 Hz.

### Reset to Raw Data
At any time, you can clear all AGC or Bandpass filters by clicking `Processing > Reset to Raw Data`.

---

## 2. Instantaneous Attributes

Complex trace attributes treat the real seismic trace as the real part of an analytical signal. The Hilbert Transform is used to compute the imaginary (quadrature) component, allowing for the extraction of instantaneous envelope, phase, and frequency.

> ⚠️ **Important Resolution Note:** Before calculating any attribute, SeisPlotPy will automatically re-read the currently visible range from disk at **Step = 1** (full resolution). This prevents attribute mathematics from running on decimated pixels, which would produce severe aliasing artifacts. The "Manual Step" checkbox in your viewport controls will be activated automatically.

### Instantaneous Amplitude (Envelope)
* **What it is:** The magnitude of the complex trace (the square root of the sum of the squares of the real and imaginary parts).
* **Best For:** Identifying bright spots, gas accumulations, sequence boundaries, and major lithological changes. It is independent of phase, so it represents the total energy reflection.

### Instantaneous Phase
* **What it is:** The phase angle of the complex trace (calculated via `arctan2`).
* **Best For:** Highlighting structural continuity, subtle faults, and pinch-outs. Because it completely ignores amplitude, every single reflector (strong or weak) is given equal visual weight.

### Cosine of Phase
* **What it is:** The cosine of the instantaneous phase.
* **Best For:** Providing a smoother, more visually comprehensible structural image than raw phase. It avoids the wraps (+180 to -180 degree jumps) inherent to raw phase plots.

### Instantaneous Frequency
* **What it is:** The time derivative of the instantaneous phase.
* **Best For:** Identifying fracture zones, fluid accumulations (which often cause a drop in high frequencies), and bed thickness variations.

### RMS Amplitude
* **What it is:** The root-mean-square amplitude computed over a sliding sample window.
* **Best For:** A smoother, more robust measure of reflection strength than Envelope, often used in direct hydrocarbon indicator (DHI) workflows.

---

## 3. Frequency Spectrum Tool

To analyze the frequency content of your data before or after applying a Bandpass filter, you can generate a spectrum plot.

1. Navigate to **Tools > Frequency Spectrum**.
2. A Matplotlib figure will open showing the amplitude spectrum.
3. The spectrum is computed using `numpy.fft.rfft` and is **averaged across all currently visible traces**.
4. The X-axis displays true **Hz**, derived dynamically from the file's binary sample interval.
