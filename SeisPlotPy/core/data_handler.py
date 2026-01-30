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

        #Fix for geographic coordinates
        self.coordinate_units = 1 # Default to 1 (Length/Meters)
        
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
            'NSummedTraces': 30,
            'NStackedTraces': 32,
            'DataUse': 34,
            'DistanceFromSourceToReceiver': 36,
            'ReceiverGroupElevation': 40,
            'SurfaceElevationAtSource': 44,
            'SourceDepthAtSurface': 48,
            'DatumElevationAtReceiver': 52,
            'DatumElevationAtSource': 56,
            'WaterDepthAtSource': 60,
            'WaterDepthAtGroup': 64,
            'SourceGroupScalar': 70,
            'ElevationScalar': 68,
            'SourceX': 72,
            'SourceY': 76,
            'GroupX': 80,
            'GroupY': 84,
            'CoordinateUnits': 88,
            'WeatheringVelocity': 90,
            'SubWeatheringVelocity': 92,
            'SourceUpholeTime': 94,
            'GroupUpholeTime': 96,
            'DelayRecordingTime': 108,
            'SamplesPerTrace': 114,
            'SampleInterval': 116,
            'CDP_X': 180,
            'CDP_Y': 184,
            'Inline': 188,
            'Crossline': 192
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

                # --- Fix: Read Coordinate Units from 1st trace ---
                if 'CoordinateUnits' in segyio.tracefield.keys:
                    self.coordinate_units = f.header[0][segyio.tracefield.keys['CoordinateUnits']]

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

            # Fix: Geographic Coordinates
            f.seek(3600 + 88)
            unit_bytes = f.read(2)
            self.coordinate_units = struct.unpack(f'{self._endian}h', unit_bytes)[0]

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
            final_type = f'{self._endian}i2' if is_short else f'{self._endian}i4'
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
    
    def get_binary_header(self):
        """Retrieves the 400-byte Binary File Header with full spec compliance."""
        binary_values = {}
        
        # Corrected field map with proper 0-indexed byte offsets
        field_map = {
            'Job ID Number': (segyio.BinField.JobID, 0, 'i'),
            'Line Number': (segyio.BinField.LineNumber, 4, 'i'),
            'Reel Number': (segyio.BinField.ReelNumber, 8, 'i'),
            'Traces per Ensemble': (segyio.BinField.Traces, 12, 'h'),
            'Aux Traces per Ensemble': (segyio.BinField.AuxTraces, 14, 'h'),
            'Sample Interval (us)': (segyio.BinField.Interval, 16, 'h'),
            'Sample Interval Original (us)': (segyio.BinField.IntervalOriginal, 18, 'h'),
            'Samples per Trace': (segyio.BinField.Samples, 20, 'h'),
            'Samples per Trace Original': (segyio.BinField.SamplesOriginal, 22, 'h'),
            'Data Sample Format Code': (segyio.BinField.Format, 24, 'h'),
            'Ensemble Fold': (segyio.BinField.EnsembleFold, 26, 'h'),
            'Trace Sorting Code': (segyio.BinField.SortingCode, 28, 'h'),
            'Vertical Sum Code': (segyio.BinField.VerticalSum, 30, 'h'),
            'Sweep Frequency Start (Hz)': (segyio.BinField.SweepFrequencyStart, 32, 'h'),
            'Sweep Frequency End (Hz)': (segyio.BinField.SweepFrequencyEnd, 34, 'h'),
            'Sweep Length (ms)': (segyio.BinField.SweepLength, 36, 'h'),
            'Sweep Type Code': (segyio.BinField.Sweep, 38, 'h'),
            'Sweep Channel': (segyio.BinField.SweepChannel, 40, 'h'),
            'Sweep Taper Start (ms)': (segyio.BinField.SweepTaperStart, 42, 'h'),
            'Sweep Taper End (ms)': (segyio.BinField.SweepTaperEnd, 44, 'h'),
            'Taper Type Code': (segyio.BinField.Taper, 46, 'h'),
            'Binary Gain Recovery Flag': (segyio.BinField.BinaryGainRecovery, 48, 'h'),
            'Amplitude Recovery Code': (segyio.BinField.AmplitudeRecovery, 50, 'h'),
            'Measurement System': (segyio.BinField.MeasurementSystem, 54, 'h'),
            'Impulse Signal Polarity': (segyio.BinField.ImpulseSignalPolarity, 56, 'h'),
            'Vibratory Polarity': (segyio.BinField.VibratoryPolarity, 58, 'h'),
            # Extended fields (Rev 2.0) - 4-byte integers
            'Ext Traces': (segyio.BinField.ExtTraces, 60, 'i'),
            'Ext Aux Traces': (segyio.BinField.ExtAuxTraces, 64, 'i'),
            'Ext Samples': (segyio.BinField.ExtSamples, 68, 'i'),
            'Ext Samples Original': (segyio.BinField.ExtSamplesOriginal, 88, 'i'),
            'Ext Ensemble Fold': (segyio.BinField.ExtEnsembleFold, 92, 'i'),
            # Revision and Flags
            'SEG-Y Revision Number': (segyio.BinField.SEGYRevision, 300, 'h'),
            'SEG-Y Revision Minor': (segyio.BinField.SEGYRevisionMinor, 302, 'h'),
            'Fixed Length Trace Flag': (segyio.BinField.TraceFlag, 304, 'h'),
            'Extended Text Header Count': (segyio.BinField.ExtendedHeaders, 306, 'h'),
        }

        try:
            if not self._use_fallback:
                # Use segyio for standard files
                with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                    for name, (enum_key, _, _) in field_map.items():
                        try:
                            binary_values[name] = int(f.bin[enum_key])
                        except:
                            binary_values[name] = 0
            else:
                # Manual binary interpretation for non-standard files
                with open(self.file_path, 'rb') as f:
                    f.seek(3200)
                    raw_bin = f.read(400)
                    if len(raw_bin) < 400: 
                        return {}

                    # Detect endianness from Revision (bytes 300-301 of binary header)
                    rev_val = struct.unpack_from(">h", raw_bin, 300)[0]
                    endian = ">" if (0 <= rev_val <= 10) else "<"

                    for name, (_, offset, fmt) in field_map.items():
                        try:
                            val = struct.unpack_from(f"{endian}{fmt}", raw_bin, offset)[0]
                            binary_values[name] = int(val)
                        except:
                            binary_values[name] = 0
                            
        except Exception as e:
            print(f"CRITICAL: Binary Header Load Failed: {e}")
            
        return binary_values
    
    def export_segy_subset(self, output_path, start_trace, end_trace):
        """
        Creates a new SEG-Y file containing only the traces from start_trace to end_trace.
        """
        try:
            # Open the original file to read from it
            with segyio.open(self.file_path, 'r', ignore_geometry=True) as src:
                
                # 1. Create a 'spec' (a blueprint) for the new file
                # We copy the blueprint from the source file so the format matches exactly
                spec = segyio.spec()
                spec.sorting = src.sorting
                spec.format = src.format
                spec.samples = src.samples
                spec.tracecount = (end_trace - start_trace) + 1 # Calculate new file size
                
                # 2. Create the new file using that blueprint
                with segyio.create(output_path, spec) as dst:
                    
                    # Copy the Text Header (the big EBCDIC block)
                    dst.text[0] = src.text[0]
                    
                    # Copy the Binary Header (the 400-byte block)
                    dst.bin = src.bin
                    
                    # 3. Copy the Traces and their Headers one by one
                    # We loop from 'start' to 'end'
                    dst_idx = 0
                    for src_idx in range(start_trace, end_trace + 1):
                        # Copy the header row
                        dst.header[dst_idx] = src.header[src_idx]
                        
                        # --- FIX: Reset the Trace Sequence Number for the new file ---
                        # TraceSequenceFile is byte 5-8. Key in segyio is 5.
                        # We use the integer 5 directly to avoid AttributeErrors with different segyio versions.
                        dst.header[dst_idx][5] = dst_idx + 1
                        
                        # Copy the trace data (wiggle values)
                        dst.trace[dst_idx] = src.trace[src_idx]
                        
                        dst_idx += 1
                        
            return True, "Export successful."
            
        except Exception as e:
            return False, f"Export failed: {str(e)}"