import segyio
import numpy as np
import os
import struct
from qgis.PyQt.QtWidgets import QMessageBox

class SeismicDataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.n_traces = 0
        self.n_samples = 0
        self.sample_rate = 0
        self.time_axis = None
        self.available_headers = []
        
        # Fallback state
        self._use_fallback = False
        self._mmap_data = None
        self._mmap_headers = None
        self._endian = '>' # Default to Big Endian for SEG-Y
        
        # Mapping for standard byte locations (byte offset from start of 240-byte trace header)
        self._header_map = {
            'TraceSequenceLine': 0,
            'TraceSequenceFile': 4,
            'OriginalFieldRecord': 8,
            'TraceNumber': 12,
            'EnergySourcePoint': 16,
            'CDP': 20,
            'TraceIdentificationCode': 28,
            'SourceX': 72,
            'SourceY': 76,
            'GroupX': 80,
            'GroupY': 84,
            'CoordinateUnits': 88,
            'CDP_X': 180,
            'CDP_Y': 184,
            'Inline': 188,
            'Crossline': 192,
            'SourceGroupScalar': 70
        }

        self._scan_file()

    def _scan_file(self):
        """Try segyio first, fallback to numpy if it fails."""
        try:
            # Attempt Standard Load
            with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                self.n_traces = f.tracecount
                self.n_samples = f.samples.size
                self.sample_rate = segyio.tools.dt(f) / 1000 
                self.time_axis = f.samples
                if segyio.tracefield.keys:
                    self.available_headers = list(segyio.tracefield.keys.keys())
                self._use_fallback = False
                
        except Exception as e:
            print(f"SeisPlotPy: Standard load failed ({e}). Asking user for fallback...")
            
            # --- CONFIRMATION DIALOG ---
            # Replicates the error message style but asks for permission to proceed
            reply = QMessageBox.question(
                None, 
                "SEG-Y Load Error", 
                f"Standard load failed: {str(e)}\n\n"
                "Do you want to proceed with a raw fallback load?\n"
                "(This ignores strict geometry checks but may take longer)",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self._scan_file_fallback()
            else:
                # Re-raise the exception so the Controller knows to stop loading
                raise Exception("User cancelled fallback load.")

    def _scan_file_fallback(self):
        """Robust reader using numpy memmap for files with broken binary headers."""
        self._use_fallback = True
        file_size = os.path.getsize(self.file_path)
        
        # 1. Read Binary Header to guess Endianness and Sample Format
        with open(self.file_path, 'rb') as f:
            f.seek(3224) # Format code location
            fmt_code = int.from_bytes(f.read(2), 'big')
            
            if fmt_code > 255: 
                self._endian = '<' # Little Endian
            else:
                self._endian = '>' # Big Endian
                
            # 2. Read First Trace Header to find REAL number of samples
            f.seek(3600 + 114) # Number of samples is at byte 115-116 of trace header
            ns_bytes = f.read(2)
            self.n_samples = struct.unpack(f'{self._endian}H', ns_bytes)[0]
            
            f.seek(3600 + 116) # Sample interval
            dt_bytes = f.read(2)
            dt_us = struct.unpack(f'{self._endian}H', dt_bytes)[0]
            self.sample_rate = dt_us / 1000.0

        # 3. Calculate Trace Count based on File Size
        trace_block_size = 240 + self.n_samples * 4
        data_size = file_size - 3600
        self.n_traces = int(data_size / trace_block_size)
        
        # 4. Create Time Axis
        self.time_axis = np.arange(self.n_samples) * self.sample_rate
        
        # 5. Setup Available Headers (Static list for fallback)
        self.available_headers = list(self._header_map.keys())
        
        # 6. Initialize Memmap
        # Assume IEEE float (f4). If data is IBM float, values will be incorrect but won't crash.
        dt_str = f'{self._endian}f4'
        dtype = np.dtype([
            ('header', np.void, 240),
            ('data', dt_str, (self.n_samples,))
        ])
        
        self._mmap_data = np.memmap(
            self.file_path, 
            dtype=dtype, 
            mode='r', 
            offset=3600,
            shape=(self.n_traces,)
        )
        print(f"SeisPlotPy: Fallback load successful. Traces: {self.n_traces}, Samples: {self.n_samples}")

    def get_data_slice(self, start_trace, end_trace, step=1):
        """Reads data traces"""
        start = max(0, start_trace)
        end = min(self.n_traces, end_trace)
        if start >= end:
            return np.zeros((self.n_samples, 0))

        if not self._use_fallback:
            with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                data_chunk = f.trace.raw[start:end:step]
                return data_chunk.T
        else:
            # Fallback Memmap Read
            chunk = self._mmap_data['data'][start:end:step]
            return chunk.T

    def get_header_slice(self, header_name, start_trace, end_trace, step=1):
        """Reads a specific header array"""
        start = max(0, start_trace)
        end = min(self.n_traces, end_trace)
        
        if not self._use_fallback:
            if header_name not in segyio.tracefield.keys:
                return np.arange(start_trace, end_trace, step)

            key = segyio.tracefield.keys[header_name]
            with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                all_values = f.attributes(key)[:]
                return all_values[start:end:step]
        else:
            # Fallback Read
            if header_name not in self._header_map:
                if header_name == "Trace Index":
                    return np.arange(start, end, step)
                return np.zeros((end-start)//step) 

            offset = self._header_map[header_name]
            
            # Determine type (Short vs Integer)
            is_short = header_name in ['SourceGroupScalar', 'CoordinateUnits', 'TraceIdentificationCode']
            dtype_code = 'h' if is_short else 'i'
            byte_len = 2 if is_short else 4
            
            # Efficient slicing from structured array headers
            headers_raw = self._mmap_data['header'][start:end:step]
            
            # Safety check
            if offset + byte_len > 240: return np.zeros(len(headers_raw))
            
            # View extraction logic
            view_u8 = np.frombuffer(headers_raw.tobytes(), dtype=np.uint8)
            view_2d = view_u8.reshape((len(headers_raw), 240))
            cols = view_2d[:, offset:offset+byte_len]
            final_type = f'{self._endian}{dtype_code}2' if is_short else f'{self._endian}{dtype_code}4'
            values = np.frombuffer(cols.tobytes(), dtype=final_type)
            
            return values

    def get_text_header(self):
        """Reads and decodes the EBCDIC/ASCII text header properly"""
        try:
            with open(self.file_path, 'rb') as f:
                raw_text = f.read(3200)
                
            is_ebcdic = False
            if len(raw_text) > 0 and raw_text[0] == 0xC3: 
                is_ebcdic = True
            
            try:
                if is_ebcdic:
                    text_str = raw_text.decode('ebcdic-cp-be')
                else:
                    text_str = raw_text.decode('ascii', errors='ignore')
            except:
                text_str = raw_text.decode('ascii', errors='ignore')

            if len(text_str) >= 3200 and '\n' not in text_str:
                    lines = [text_str[i:i+80] for i in range(0, len(text_str), 80)]
                    return "\n".join(lines)
            
            return text_str
                
        except Exception as e:
            return f"Error reading text header: {e}"