import segyio
import numpy as np

class SeismicDataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.n_traces = 0
        self.n_samples = 0
        self.sample_rate = 0
        self.time_axis = None
        
        # Scan available headers immediately
        self.available_headers = []
        if segyio.tracefield.keys:
            self.available_headers = list(segyio.tracefield.keys.keys())
        
        self._scan_file()

    def _scan_file(self):
        with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
            self.n_traces = f.tracecount
            self.n_samples = f.samples.size
            self.sample_rate = segyio.tools.dt(f) / 1000 
            self.time_axis = f.samples

    def get_data_slice(self, start_trace, end_trace, step=1):
        """Reads data traces"""
        start = max(0, start_trace)
        end = min(self.n_traces, end_trace)
        if start >= end:
            return np.zeros((self.n_samples, 0))

        with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
            data_chunk = f.trace.raw[start:end:step]
            return data_chunk.T

    def get_header_slice(self, header_name, start_trace, end_trace, step=1):
        """
        Reads a specific header array (e.g., CDP) for the requested range.
        """
        if header_name not in segyio.tracefield.keys:
            return np.arange(start_trace, end_trace, step)

        key = segyio.tracefield.keys[header_name]
        start = max(0, start_trace)
        end = min(self.n_traces, end_trace)

        with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
            all_values = f.attributes(key)[:]
            return all_values[start:end:step]

    def get_text_header(self):
        """Reads and decodes the EBCDIC/ASCII text header properly"""
        try:
            with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                raw_text = f.text[0]
                
                # 1. Convert bytearray to immutable bytes for safer handling
                if isinstance(raw_text, bytearray):
                    raw_text = bytes(raw_text)
                
                # 2. Smart Decode: Check for EBCDIC signature
                # EBCDIC 'C' is 0xC3 (195). ASCII 'C' is 0x43 (67).
                # Standard SEG-Y headers start with "C 1 CLIENT..."
                is_ebcdic = False
                if len(raw_text) > 0 and raw_text[0] == 0xC3:
                    is_ebcdic = True
                
                # 3. Decode
                try:
                    if is_ebcdic:
                        text_str = raw_text.decode('ebcdic-cp-be')
                    else:
                        text_str = raw_text.decode('ascii', errors='ignore')
                except Exception:
                    # Fallback if detection failed
                    text_str = raw_text.decode('ascii', errors='ignore')

                # 4. Format Logic
                # If the string is a solid block of 3200 chars (Standard Tape Format),
                # Wrap it every 80 characters for readability.
                # If it already has newlines (Desktop SEG-Y), leave it alone.
                if len(text_str) >= 3200 and '\n' not in text_str:
                     lines = [text_str[i:i+80] for i in range(0, len(text_str), 80)]
                     return "\n".join(lines)
                
                return text_str
                
        except Exception as e:
            return f"Error reading text header: {e}"