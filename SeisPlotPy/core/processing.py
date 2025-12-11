import numpy as np
from scipy.signal import butter, filtfilt, hilbert

class SeismicProcessing:
    @staticmethod
    def apply_scalar(values, scalars):
        if scalars is None: return values.astype(float)
        s = scalars.astype(float); s[s == 0] = 1.0
        res = values.astype(float)
        pos_mask = s > 0; res[pos_mask] *= s[pos_mask]
        neg_mask = s < 0; res[neg_mask] /= np.abs(s[neg_mask])
        return res

    @staticmethod
    def calculate_cumulative_distance(x, y):
        dx = np.diff(x); dy = np.diff(y)
        dist = np.sqrt(dx**2 + dy**2)
        return np.concatenate(([0], np.cumsum(dist)))

    @staticmethod
    def apply_agc(data, sample_rate_ms, window_ms=500):
        epsilon = 1e-10
        window_len = int(window_ms / sample_rate_ms)
        if window_len % 2 == 0: window_len += 1
        squared = data ** 2
        window = np.ones(window_len) / window_len
        n_samples, n_traces = data.shape
        data_agc = np.zeros_like(data)
        for i in range(n_traces):
            trace_sq = squared[:, i]
            rms_env = np.sqrt(np.convolve(trace_sq, window, mode='same'))
            data_agc[:, i] = data[:, i] / (rms_env + epsilon)
        return data_agc

    @staticmethod
    def apply_bandpass(data, sample_rate_ms, lowcut, highcut, order=4):
        nyq = 0.5 * (1000.0 / sample_rate_ms)
        low = lowcut / nyq; high = highcut / nyq
        if low <= 0: low = 0.001
        if high >= 1: high = 0.99
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    @staticmethod
    def calculate_spectrum(data, sample_rate_ms):
        n_samples = data.shape[0]
        fft_vals = np.fft.rfft(data, axis=0)
        fft_mag = np.abs(fft_vals)
        avg_spectrum = np.mean(fft_mag, axis=1)
        dt_sec = sample_rate_ms / 1000.0
        freqs = np.fft.rfftfreq(n_samples, d=dt_sec)
        return freqs, avg_spectrum

    # --- ATTRIBUTE METHODS ---

    @staticmethod
    def attribute_envelope(data):
        """Instantaneous Amplitude (Reflection Strength)"""
        # Hilbert transform along time axis (axis=0)
        analytic_signal = hilbert(data, axis=0)
        return np.abs(analytic_signal)

    @staticmethod
    def attribute_phase(data):
        """Instantaneous Phase (-pi to pi)"""
        analytic_signal = hilbert(data, axis=0)
        return np.angle(analytic_signal)

    @staticmethod
    def attribute_frequency(data, sample_rate_ms):
        """Instantaneous Frequency (Hz)"""
        analytic_signal = hilbert(data, axis=0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=0)
        # Frequency is derivative of phase / (2*pi)
        # diff reduces length by 1, so append 0 to keep shape
        freq = np.diff(instantaneous_phase, axis=0) / (2.0 * np.pi) * (1000.0 / sample_rate_ms)
        return np.vstack((freq, freq[-1, :])) # Pad last sample

    @staticmethod
    def attribute_cosine_phase(data):
        """Cosine of Instantaneous Phase (Good for continuity)"""
        analytic_signal = hilbert(data, axis=0)
        return np.cos(np.angle(analytic_signal))
        
    @staticmethod
    def attribute_rms(data, sample_rate_ms, window_ms=100):
        """Root Mean Square Amplitude"""
        window_len = int(window_ms / sample_rate_ms)
        if window_len % 2 == 0: window_len += 1
        squared = data ** 2
        window = np.ones(window_len) / window_len
        rms = np.zeros_like(data)
        for i in range(data.shape[1]):
            rms[:, i] = np.sqrt(np.convolve(squared[:, i], window, mode='same'))
        return rms