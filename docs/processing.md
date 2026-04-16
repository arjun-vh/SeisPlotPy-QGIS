# Seismic Processing & Attributes

SeisPlotPy includes a suite of real-time digital signal processing tools. These functions allow you to enhance reflectors, balance amplitudes, and extract instantaneous attributes directly within the viewer without needing to permanently modify your source SEG-Y file.

All processing tools can be accessed via the **Processing** and **Attributes** menus in the top menu bar.

---

## 1. Basic Processing Tools

These tools alter the amplitude and frequency content of the currently viewed seismic data to improve interpretation visibility.

### Apply AGC (Automatic Gain Control)
Seismic data often suffers from amplitude decay over time/depth. AGC balances these amplitudes by applying a sliding scaling window.
1. Navigate to **Processing > Apply AGC**.
2. A prompt will ask for a **Window Size (ms)**. The default is 500 ms.
3. A smaller window (e.g., 200 ms) will equalize amplitudes aggressively, while a larger window (e.g., 1000 ms) preserves more of the relative true amplitude dynamics. 

### Bandpass Filter
Remove unwanted high-frequency noise or low-frequency swell by applying a Butterworth bandpass filter.
1. Navigate to **Processing > Bandpass Filter**.
2. The tool automatically calculates the Nyquist limit based on your file's sample rate.
3. Enter your desired **Low Cut** and **High Cut** frequencies (in Hz).
4. The filter is applied instantaneously to the visible data.

### Reset to Raw Data
If you have applied AGC or a filter and want to return to the original amplitude values, simply click **Processing > Reset to Raw Data**. This will reload the current trace slice directly from the SEG-Y file.

---

## 2. Instantaneous Seismic Attributes

Seismic attributes extract hidden information from the complex trace (using the Hilbert Transform) to help identify sequence boundaries, gas masking, and structural continuity.

> ⚠️ **Important Mathematical Note:** To ensure mathematical accuracy, SeisPlotPy **will automatically reload your current view at full resolution (Trace Step = 1)** before calculating an attribute. Attributes calculated on decimated data are unreliable, so the plugin prevents this automatically. If you are viewing a massive section of the line, this full-resolution fetch may take a few moments.

To apply an attribute, navigate to the **Attributes** menu:

* **Instantaneous Amplitude (Envelope):** Represents the reflection strength. Excellent for identifying gas accumulations, major sequence boundaries, and tuning effects.
* **Instantaneous Phase:** Measures the continuity of events independently of amplitude.
* **Cosine of Phase:** Similar to Instantaneous Phase, but without the abrupt $-pi$ to $+pi$ wraparound discontinuities. Highly recommended for tracing faults and continuous horizons.
* **Instantaneous Frequency:** The rate of change of the phase. Useful for identifying attenuation (e.g., beneath gas sands) and bed thickness variations.
* **RMS Amplitude:** Calculates the Root Mean Square amplitude over a user-defined sliding window (in ms). Excellent for identifying isolated high-amplitude anomalies over a background trend.

*To clear an attribute and return to standard structural viewing, use **Processing > Reset to Raw Data**.*

---

## 3. Tool: Frequency Spectrum

If you need to analyze the frequency content of your data (for example, to decide on Bandpass Filter parameters), you can generate a spectrum plot.

1. Zoom in on a specific region of interest in the viewer.
2. Navigate to **Tools > Frequency Spectrum**.
3. A new window will appear showing the amplitude spectrum calculated from the currently visible data array, allowing you to easily identify the dominant frequency and any high/low-frequency noise spikes.
