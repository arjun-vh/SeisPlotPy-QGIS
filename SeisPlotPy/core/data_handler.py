import segyio
import numpy as np
import os
import struct
import re
from qgis.PyQt.QtWidgets import QMessageBox

class SeismicDataManager:
    def __init__(self, file_path, suppress_dialogs=False):
        self.file_path = file_path
        self.suppress_dialogs = suppress_dialogs
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
    
    # Corrected field map with proper 0-indexed byte offsets
    # Format: 'Friendly Name': (segyio_enum_key, byte_offset, struct_format_char)
    BINARY_FIELD_MAP = {
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

    # Standard Trace Header Map (SEG-Y Rev 1)
    # Format: 'Name': (ByteOffset (0-based from start of 240), Type ('i'=4-byte, 'h'=2-byte, 'H'=2-byte unsigned))
    TRACE_FIELD_MAP = {
        'TraceSequenceLine': (0, 'i'),
        'TraceSequenceFile': (4, 'i'),
        'OriginalFieldRecord': (8, 'i'),
        'TraceNumber': (12, 'i'),
        'EnergySourcePoint': (16, 'i'),
        'CDP': (20, 'i'),
        'TraceSequenceEnsemble': (24, 'i'),
        'TraceIdentificationCode': (28, 'h'), # 29-30
        'NSummedTraces': (30, 'h'), # 31-32
        'NStackedTraces': (32, 'h'), # 33-34
        'DataUse': (34, 'h'), # 35-36
        'DistanceFromSourceToReceiver': (36, 'i'),
        'ReceiverGroupElevation': (40, 'i'),
        'SurfaceElevationAtSource': (44, 'i'),
        'SourceDepthAtSurface': (48, 'i'),
        'DatumElevationAtReceiver': (52, 'i'),
        'DatumElevationAtSource': (56, 'i'),
        'WaterDepthAtSource': (60, 'i'),
        'WaterDepthAtGroup': (64, 'i'),
        'ElevationScalar': (68, 'h'), # 69-70
        'SourceGroupScalar': (70, 'h'), # 71-72
        'SourceX': (72, 'i'),
        'SourceY': (76, 'i'),
        'GroupX': (80, 'i'),
        'GroupY': (84, 'i'),
        'CoordinateUnits': (88, 'h'), # 89-90
        'WeatheringVelocity': (90, 'h'),
        'SubWeatheringVelocity': (92, 'h'),
        'SourceUpholeTime': (94, 'h'),
        'GroupUpholeTime': (96, 'h'),
        'SourceStaticCorrection': (98, 'h'),
        'GroupStaticCorrection': (100, 'h'),
        'TotalStaticApplied': (102, 'h'),
        'LagTimeA': (104, 'h'),
        'LagTimeB': (106, 'h'),
        'DelayRecordingTime': (108, 'h'),
        'MuteTimeStart': (110, 'h'),
        'MuteTimeEnd': (112, 'h'),
        'SamplesPerTrace': (114, 'H'), # Unsigned
        'SampleInterval': (116, 'H'), # Unsigned
        'GainType': (118, 'h'),
        'InstrumentGainConstant': (120, 'h'),
        'InstrumentInitialGain': (122, 'h'),
        'Correlated': (124, 'h'),
        'SweepFrequencyStart': (126, 'h'),
        'SweepFrequencyEnd': (128, 'h'),
        'SweepLength': (130, 'h'),
        'SweepType': (132, 'h'),
        'SweepTraceTaperLengthStart': (134, 'h'),
        'SweepTraceTaperLengthEnd': (136, 'h'),
        'TaperType': (138, 'h'),
        'AliasFilterFrequency': (140, 'h'),
        'AliasFilterSlope': (142, 'h'),
        'NotchFilterFrequency': (144, 'h'),
        'NotchFilterSlope': (146, 'h'),
        'LowCutFrequency': (148, 'h'),
        'HighCutFrequency': (150, 'h'),
        'LowCutSlope': (152, 'h'),
        'HighCutSlope': (154, 'h'),
        'YearDataRecorded': (156, 'h'),
        'DayOfYear': (158, 'h'),
        'HourOfDay': (160, 'h'),
        'MinuteOfHour': (162, 'h'),
        'SecondOfMinute': (164, 'h'),
        'TimeBasisCode': (166, 'h'),
        'TraceWeightingFactor': (168, 'h'),
        'GeophoneGroupNumberRoll1': (170, 'h'),
        'GeophoneGroupTraceNumber1': (172, 'h'),
        'GeophoneGroupNumberLastTrace': (174, 'h'),
        'GeophoneGroupTraceNumberLast': (176, 'h'),
        'GapSize': (178, 'h'),
        'TaperOvertravel': (180, 'h'),
        'CDP_X': (180, 'i'), # NOTE: Overlaps with GapSize/Taper depending on revision. We prioritize standard CDP headers.
        'CDP_Y': (184, 'i'),
        'Inline': (188, 'i'),
        'Crossline': (192, 'i'),
        'ShotPoint': (196, 'i'),
        'ShotPointScalar': (200, 'h'),
        'TraceValueMeasurementUnit': (202, 'h'),
        'TransductionConstantMandissa': (204, 'i'),
        'TransductionConstantPower': (208, 'h'),
        'TransductionUnit': (210, 'h'),
        'TraceIdentifier': (212, 'h'),
        'ScalarTraceHeader': (214, 'h'),
        'SourceType': (216, 'h'),
        'SourceEnergyDirectionMantissa': (218, 'i'),
        'SourceEnergyDirectionExponent': (222, 'h'),
        'SourceMeasurementMantissa': (224, 'i'),
        'SourceMeasurementExponent': (228, 'h'),
        'SourceMeasurementUnit': (230, 'h'),
    }

    def _scan_file(self):
        """Try segyio first, fallback to numpy if it fails."""
        try:
            # Attempt Standard Load
            with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                # Use heuristic scan even for standard load to provide independent verification
                # We need to close the file or read separately, as segyio has it open.
                # Actually _infer_format_from_data opens its own handle, which is fine (OS allows shared read)
                # self._detected_format = self._infer_format_from_data()
                
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
            
            if self.suppress_dialogs:
                # Silently attempt fallback during batch processing
                self._scan_file_fallback()
                return
                
            # --- CONFIRMATION DIALOG ---
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
        
        # 1. Read Binary Header to guess Endianness 
        with open(self.file_path, 'rb') as f:
            f.seek(3224) 
            # Format Code is at 3224 (0-based 3200 + 24)
            fmt_code = int.from_bytes(f.read(2), 'big')
            if fmt_code > 255: 
                self._endian = '<' # Little Endian
            else:
                self._endian = '>' # Big Endian
                
            # --- FIX: Detect Data Sample Format ---
            # Format 1 = IBM Float, Format 2 = Int32, Format 3 = Int16, Format 5 = IEEE Float
            # We need to re-read carefully
            f.seek(3200 + 24)
            fmt_bytes = f.read(2)
            self._detected_fmt_code = struct.unpack(f'{self._endian}h', fmt_bytes)[0]
            print(f"SeisPlotPy: Fallback detected format code: {self._detected_fmt_code} (Endian: {self._endian})")

        # 2. Read First Trace Header to find REAL number of samples
        with open(self.file_path, 'rb') as f:
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
        # 6. Initialize Memmap
        # Check Format Code
        # Check Format Code
        # Check Format Code
        # FIX: User report indicates some files have Format 1 (IBM) but contain IEEE data.
        # verifying "Broken Header" hypothesis: In fallback mode, prefer IEEE unless assumed otherwise.
        # if self._detected_fmt_code == 1:
             # dt_str = '>u4' 
        # 6. Initialize Memmap
       
        # FIX: Heuristic Auto-Detection for Float Data
        # Problem: Some files claim IBM Float (Code 1) but have IEEE data. Others are valid IBM.
        # Solution: Test 1st trace with IEEE. If it looks "sane", keep IEEE. If garbage, try IBM.
        
        detected_dt_str = None
        
        # Only perform heuristic for potentially float data (Format 1 or 5 or unknown)
        if self._detected_fmt_code not in [2, 3]:
            try:
                # A. Test as IEEE Float (Most common modern format)
                test_dt = np.dtype(f'{self._endian}f4')
                with open(self.file_path, 'rb') as f:
                    f.seek(3600 + 240) # Jump to end of first trace header
                    raw_bytes = f.read(self.n_samples * 4)
                    
                test_data = np.frombuffer(raw_bytes, dtype=test_dt)
                
                # Check 1: Range (Common seismic amps are < 1e10. Garbage floats are often 1e38 or NaN)
                max_val = np.max(np.abs(test_data))
                has_bad_vals = np.isnan(max_val) or np.isinf(max_val) or max_val > 1e20
                
                # Check 2: Variance/Signal (Garbage often has extremely low or high variance)
                # If IEEE looks "sane", we prioritize it even if header says IBM
                if not has_bad_vals:
                    print(f"SeisPlotPy: Heuristic - IEEE Float looks valid (Max: {max_val:.2e}). Using IEEE.")
                    detected_dt_str = f'{self._endian}f4'
                else:
                    print(f"SeisPlotPy: Heuristic - IEEE Float looks INVALID (Max: {max_val}). Falling back to IBM check.")
            
            except Exception as e:
                 print(f"SeisPlotPy: Heuristic check failed: {e}")

        # Final Decision Logic
        # FIX: Introduce _use_ibm_conversion flag to decouple format detection from conversion
        self._use_ibm_conversion = False
        
        if detected_dt_str:
            dt_str = detected_dt_str
            # Heuristic picked IEEE, do NOT convert as IBM
            self._use_ibm_conversion = False
        elif self._detected_fmt_code == 1:
            # IEEE failed or wasn't tried, and header explicitly says IBM -> Use IBM
            dt_str = '>u4' # Force Big Endian for IBM conversion
            self._use_ibm_conversion = True
            print("SeisPlotPy: Fallback using IBM Float (Header Code 1)")
        elif self._detected_fmt_code == 2:
            dt_str = f'{self._endian}i4'
        elif self._detected_fmt_code == 3:
            dt_str = f'{self._endian}i2'
        else:
            # Default fallback
            dt_str = f'{self._endian}f4'
            
        print(f"SeisPlotPy: Fallback selected data type: {dt_str}, IBM Conversion: {self._use_ibm_conversion}")

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
            
            # --- FIX: IBM Float Conversion (uses heuristic flag) ---
            if self._use_ibm_conversion:
                chunk = self._convert_ibm_to_ieee(chunk)
                
            return chunk.T

    def _convert_ibm_to_ieee(self, ibm_data):
        """
        Converts IBM floating point (uint32 view) to IEEE float32.
        Vectorized numpy implementation for performance.
        Source: Public domain / Common seismic algorithms
        """
        # 1. Sign bit (Mask 0x80000000)
        sign = np.bitwise_and(ibm_data, 0x80000000) >> 31
        sign = (-1.0) ** sign
        
        # 2. Exponent (Mask 0x7F000000, 7 bits, base 16 excess 64)
        exponent = np.bitwise_and(ibm_data, 0x7F000000) >> 24
        # IBM = 16 ^ (exp - 64) = 2 ^ (4 * (exp - 64))
        # We need efficient power calc. 
        # Using floating point pow is cleanest
        exponent = (exponent - 64) * 4.0 
        
        # 3. Mantissa (Mask 0x00FFFFFF, 24 bits, normalized differently)
        mantissa = np.bitwise_and(ibm_data, 0x00FFFFFF)
        # IBM Mantissa is fraction 0 <= m < 1. Value = m * 16^exp
        # The integer 'mantissa' here effectively needs to be divided by 2^24 to be a fraction?
        # No, byte layout issues.
        # Actually: M / 2^24
        mantissa = mantissa.astype(np.float32) / (2**24)
        
        # Result = Sign * Mantissa * 2^Exponent
        return sign * mantissa * (2.0 ** exponent)

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
            except Exception:
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
        
        try:
            if not self._use_fallback:
                # Use segyio for standard files
                with segyio.open(self.file_path, mode='r', ignore_geometry=True) as f:
                    for name, (enum_key, _, _) in self.BINARY_FIELD_MAP.items():
                        try:
                            binary_values[name] = int(f.bin[enum_key])
                        except Exception:
                            binary_values[name] = 0
            else:
                # Manual binary interpretation for non-standard files
                with open(self.file_path, 'rb') as f:
                    f.seek(3200)
                    raw_bin = f.read(400)
                    if len(raw_bin) < 400: 
                        return {}

                    
                    # --- FIX: Use detectors from _scan_file_fallback ---
                    # Don't try to guess again, use what we already know works
                    endian = self._endian

                    for name, (_, offset, fmt) in self.BINARY_FIELD_MAP.items():
                        try:
                            val = struct.unpack_from(f"{endian}{fmt}", raw_bin, offset)[0]
                            binary_values[name] = int(val)
                        except Exception:
                            binary_values[name] = 0
                            
        except Exception as e:
            print(f"CRITICAL: Binary Header Load Failed: {e}")
            
        return binary_values

    def _sanitize_text_header(self, text):
        """
        Parses raw text (potentially with 'C 1' prefixes) and returns a standard 
        segyio-friendly dictionary for header creation.
        Ensures strictly 40 lines, max 75 chars content per line.
        """
        lines = text.split('\n')
        clean_lines = {}
        
        for i in range(40):
            content = ""
            if i < len(lines):
                raw_line = lines[i].strip()
                # Remove existing 'C 1' or 'C01' prefixes if present
                # Regex heuristic: Starts with C, optional space, digits, space
                # Match "C 1 " or "C01 " or "C1 " at start
                match = re.match(r'^C\s*\d+\s+(.*)', raw_line)
                if match:
                    content = match.group(1)
                else:
                    # Maybe the user deleted the C-number entirely and just wrote text
                    # Check if it looks like a prefix "C 1" without text
                    if re.match(r'^C\s*\d+\s*$', raw_line):
                        content = ""
                    else:
                        content = raw_line
            
            # Truncate to ensure it fits (80 chars total - 5 chars for "C 1 ") = 75 safe
            # Also sanitize characters that might break EBCDIC (like | or tabs)
            content = content.replace('\t', '    ')
            content = content.replace('|', ':') # Replace pipe with colon for safety
            
            # Force ASCII to avoid weird unicode issues
            content = content.encode('ascii', 'replace').decode('ascii')
            
            clean_lines[i+1] = content[:75]
            
        return clean_lines

    def save_new_segy_with_header(self, output_path, new_header_values, progress_callback=None):
        """
        Creates a new SEG-Y file with modified binary header values.
        Supports both standard segyio-loaded files and 'fallback' raw-loaded files.
        
        Args:
            output_path: Path to write the new file.
            new_header_values: Dictionary of {'Field Name': new_int_value}.
        """
        try:
            if not self._use_fallback:
                # --- SCENARIO A: Standard segyio file ---
                # Safe to just use segyio.create, it defaults to Big Endian IEEE (usually)
                # But to be safe, we explicitly set spec format
                with segyio.open(self.file_path, 'r', ignore_geometry=True) as src:
                    # Create spec based on source
                    spec = segyio.spec()
                    spec.sorting = src.sorting
                    spec.format = 5 # FORCE IEEE Float (Big Endian standard)
                    spec.samples = src.samples
                    spec.tracecount = src.tracecount
                    
                    with segyio.create(output_path, spec) as dst:
                        # 1. Copy Text Header (No endianness, just chars)
                        dst.text[0] = src.text[0]
                        
                        # 2. Copy Binary Header
                        dst.bin = src.bin
                        # FORCE Format Code 5 (IEEE) in binary header
                        dst.bin[segyio.BinField.Format] = 5
                        
                        # 3. Apply User Updates to Binary Header
                        for name, val in new_header_values.items():
                            if name in self.BINARY_FIELD_MAP:
                                enum_key = self.BINARY_FIELD_MAP[name][0]
                                dst.bin[enum_key] = int(val)
                                
                        # 4. Copy Traces
                        # segyio handles data conversion automatically if spec.format is different
                        # We copy in blocks to be efficient
                        n_traces = src.tracecount
                        block_size = 1000
                        
                        for i in range(0, n_traces, block_size):
                            count = min(block_size, n_traces - i)
                            dst.header[i : i+count] = src.header[i : i+count]
                            # Copy trace data (numpy array) - segyio writes it as Big Endian IEEE
                            dst.trace[i : i+count] = src.trace[i : i+count]
                            
                            if progress_callback:
                                progress_callback(int((i / n_traces) * 100))
                                
            else:
                # --- SCENARIO B: Fallback / Broken File (Raw Patching + Conversion) ---
                # We interpret input using self._endian, output as Big Endian (>)
                
                # 1. Read the original 3600-byte header block
                with open(self.file_path, 'rb') as f_in:
                    header_block = f_in.read(3600)
                    
                    # 2. Convert Binary Header to Big Endian
                    # The binary header is bytes 3200-3600
                    # We repack existing values into Big Endian
                    bin_header = bytearray(header_block[3200:])
                    
                    # ... Wait, simply repacking user updates is not enough. 
                    # We might need to flip the entire header if original was Little Endian.
                    # Simplest approach: Use our mapping to read ALL known fields and write them back as BE.
                    
                    # Let's create a clean 400-byte buffer
                    new_bin_header = bytearray(400)
                    
                    # Copy 3200-3600 from source first (to keep unknown fields)
                    # But if endianness is swapped, unknown fields will be garbage.
                    # User accepted this is a "Repair". We keep it simple.
                    
                    # If input is Little Endian, we MUST swap bytes of the 400-byte block first?
                    # No, let's just write the fields we know.
                    # Actually, if we just swap 4-byte and 2-byte words, we cover most things.
                    # But specific fields are safer.
                    
                    # Better Strategy: 
                    # 1. Use existing 'binary_values' logic to READ everything current
                    # 2. Update with user values
                    # 3. Pack everything as Big Endian
                    
                    # (Quick hack: we just patch the user-provided values and Force Format=5,
                    #  and assume the rest of the file is readable enough?
                    #  No, user wants "Standard Output". We must swap the whole binary header if source was LE)
                    
                    if self._endian == '<':
                         # Brute force swap of the buffer? No, mixed 2 and 4 byte fields.
                         # We rely on our BINARY_FIELD_MAP to read/write known fields.
                         pass
                    
                    # Update strictly known fields from map + User modifications
                    # We'll work on a copy of the buffer
                    
                    current_bin_dict = self.get_binary_header() # Uses self._endian to read correctly
                    
                    # Apply updates
                    current_bin_dict.update(new_header_values)
                    
                    # Force Format 5
                    current_bin_dict['Data Sample Format Code'] = 5
                    
                    # Re-pack using Big Endian
                    header_ba = bytearray(header_block) # Full 3600
                    
                    for name, val in current_bin_dict.items():
                        if name in self.BINARY_FIELD_MAP:
                            _, offset, fmt = self.BINARY_FIELD_MAP[name]
                            abs_offset = 3200 + offset
                            pack_fmt = f">{fmt}" # FORCE Big Endian
                            packed_val = struct.pack(pack_fmt, int(val))
                            for b_idx, b in enumerate(packed_val):
                                header_ba[abs_offset + b_idx] = b
                                
                    # 3. Write new file
                    with open(output_path, 'wb') as f_out:
                        f_out.write(header_ba) # Writes Text (Mixed) + Binary (Big Endian)
                        
                        # 4. Stream Copy + Data Conversion
                        # Input Data is self._endian float32 (f4)
                        # Output Data must be Big Endian float32 (f4)
                        
                        trace_data_len = self.n_samples * 4
                        trace_block_size = 240 + trace_data_len
                        
                        f_in.seek(3600)
                        
                        count = 0
                        while True:
                            block = f_in.read(trace_block_size)
                            if len(block) < trace_block_size: break
                            
                            trace_head = bytearray(block[:240])
                            trace_data = block[240:]
                            
                            # A. Fix Trace Header (Swap bytes if needed)
                            # We should iterate known trace headers and swap them to BE?
                            # Or just critical ones?
                            # Critical: TraceSequenceLine (0), TraceSequenceFile (4), Inline (188), Crossline (192), CDP_X/Y
                            # For now, let's swap the standard ones we map.
                            
                            if self._endian == '<':
                                # Targeted swap for headers we use
                                for h_name, (h_off, fmt_char) in self.TRACE_FIELD_MAP.items():
                                    try:
                                        # Use 'unpack_from' with original endianness, 'pack_into' with Big Endian
                                        val = struct.unpack_from(f"{self._endian}{fmt_char}", trace_head, h_off)[0]
                                        struct.pack_into(f">{fmt_char}", trace_head, h_off, val)
                                    except Exception as e:
                                        print(f"SeisPlotPy Warning [Header Endian Swap]: {e}")

                            # B. Convert Data Samples (VITAL)
                            # Input: self._endian float32
                            # Output: Big Endian float32
                            # Numpy makes this easy
                            data_arr = np.frombuffer(trace_data, dtype=f'{self._endian}f4')
                            data_be = data_arr.astype('>f4')
                            
                            f_out.write(trace_head)
                            f_out.write(data_be.tobytes())
                            
                            count += 1
                            if progress_callback and count % 1000 == 0:
                                progress_callback(int((count / self.n_traces) * 100))
                                
            return True, "File saved successfully (Standardized to Big Endian)."
            
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def export_segy_subset(self, output_path, start_trace, end_trace, text_header_override=None):
        """
        Creates a new SEG-Y file containing only the traces from start_trace to end_trace.
        """
        try:
            if not self._use_fallback:
                # --- SCENARIO A: Standard segyio file ---
                # Open the original file to read from it
                with segyio.open(self.file_path, 'r', ignore_geometry=True) as src:
                    
                    # 1. Create a 'spec' (a blueprint) for the new file
                    # We copy the blueprint from the source file so the format matches exactly
                    spec = segyio.spec()
                    spec.sorting = src.sorting
                    spec.format = 5 # FORCE IEEE Float (Big Endian)
                    spec.samples = src.samples
                    spec.tracecount = (end_trace - start_trace) + 1 # Calculate new file size
                    
                    # 2. Create the new file using that blueprint
                    with segyio.create(output_path, spec) as dst:
                        
                        # Copy the Text Header (the big EBCDIC block)
                        if text_header_override:
                             # Sanitize input to ensure valid "C xx" structure
                             clean_dict = self._sanitize_text_header(text_header_override)
                             dst.text[0] = segyio.tools.create_text_header(clean_dict)
                        else:
                             dst.text[0] = src.text[0]
                        
                        # Copy the Binary Header (the 400-byte block)
                        dst.bin = src.bin
                        # Force IEEE Float
                        dst.bin[segyio.BinField.Format] = 5
                        
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
            else:
                # --- SCENARIO B: Fallback / Broken File (Raw Copy + Conversion) ---
                # Manual subset export with Big Endian enforcement
                trace_size_bytes = 240 + (self.n_samples * 4) 
                
                with open(self.file_path, 'rb') as f_src, open(output_path, 'wb') as f_dst:
                    # 1. Copy Text + Binary Header (3600 bytes)
                    # We need to swap the binary header if input is Little Endian
                    header_block = bytearray(f_src.read(3600))
                    
                    # Handle Text Header Override
                    if text_header_override:
                         clean_dict = self._sanitize_text_header(text_header_override)
                         new_text_header = segyio.tools.create_text_header(clean_dict)
                         header_block[:3200] = new_text_header
                    
                    # Fix Binary Header (End of block)
                    # Read current values using our mapper (handles input endianness)
                    current_bin = self.get_binary_header()
                    current_bin['Data Sample Format Code'] = 5 # Force IEEE
                    
                    # Repack binary header as Big Endian
                    for name, val in current_bin.items():
                        if name in self.BINARY_FIELD_MAP:
                            _, offset, fmt = self.BINARY_FIELD_MAP[name]
                            abs_offset = 3200 + offset
                            pack_fmt = f">{fmt}" # FORCE Big Endian
                            packed_val = struct.pack(pack_fmt, int(val))
                            for b_idx, b in enumerate(packed_val):
                                header_block[abs_offset + b_idx] = b

                    f_dst.write(header_block)
                    
                    # 2. Seek to Start Trace
                    start_offset = 3600 + (start_trace * trace_size_bytes)
                    f_src.seek(start_offset)
                    
                    # 3. Stream Copy Loop
                    traces_to_copy = (end_trace - start_trace) + 1
                    
                    for i in range(traces_to_copy):
                        # Read trace block
                        block = f_src.read(trace_size_bytes)
                        if len(block) < trace_size_bytes: break
                        
                        trace_head = bytearray(block[:240])
                        trace_data = block[240:]
                        
                        # Fix TraceSequenceFile (Bytes 4-8, 1-based index) for NEW file
                        # And ensure headers are swapped if needed
                        
                        # If input was Little Endian, we should swap headers to match output (Big Endian)
                        if self._endian == '<':
                             for h_name, (h_off, fmt_char) in self.TRACE_FIELD_MAP.items():
                                    try:
                                        # Skip overlapping fields if needed, but standard map is prioritized
                                        # Use 'unpack_from' with original endianness, 'pack_into' with Big Endian
                                        val = struct.unpack_from(f"{self._endian}{fmt_char}", trace_head, h_off)[0]
                                        struct.pack_into(f">{fmt_char}", trace_head, h_off, val)
                                    except Exception as e:
                                        print(f"SeisPlotPy Warning [Export Header Endian Swap]: {e}")
                        
                        # Overwrite TraceSequenceFile (always >i encoded now)
                        struct.pack_into('>i', trace_head, 4, i + 1)
                        
                        # Convert Data Samples
                        data_arr = np.frombuffer(trace_data, dtype=f'{self._endian}f4')
                        data_be = data_arr.astype('>f4')
                        
                        f_dst.write(trace_head)
                        f_dst.write(data_be.tobytes())

            return True, "Export successful (Standardized to Big Endian)."
            
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def dump_headers_to_csv(self, output_path, headers):
        """
        Exports selected headers to a CSV file.
        headers: list of strings (e.g. ['TraceIndex', 'CDP_X', ...])
        """
        try:
            import pandas as pd
            
            data = {}
            # Handle TraceIndex first if present
            if 'TraceIndex' in headers:
                data['TraceIndex'] = np.arange(self.n_traces)
                
            # Fetch other headers
            for h in headers:
                if h == 'TraceIndex': continue
                data[h] = self.get_header_slice(h, 0, self.n_traces, 1)
                
            df = pd.DataFrame(data)
            # Ensure columns are in the requested order (for UX)
            valid_headers = [h for h in headers if h in data]
            df = df[valid_headers]
            
            df.to_csv(output_path, index=False)
            return True, f"Successfully exported {len(valid_headers)} headers to {output_path}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def patch_headers_from_csv(self, input_csv, output_segy, mapping, progress_callback=None):
        """
        Streams the original SEG-Y, updates headers from CSV, and writes new file.
        mapping: dict {csv_col_name: segy_header_key_string}
        progress_callback: function(percent_int)
        """
        try:
            import pandas as pd
            import segyio
            
            # 1. Read CSV
            df = pd.read_csv(input_csv)
            n_rows = len(df)
            
            # Validation
            if n_rows != self.n_traces:
                 return False, f"Mismatch: CSV has {n_rows} rows, but SEG-Y has {self.n_traces} traces."
            
            # Resolve keys (String -> Int/Object)
            # mapping_resolved: {csv_col: segyio_field_key}
            mapping_resolved = {}
            for csv_col, header_name in mapping.items():
                if csv_col not in df.columns:
                     return False, f"CSV missing column: {csv_col}"
                
                # Use standard segyio mapping first, then fallback
                if header_name in segyio.tracefield.keys:
                     mapping_resolved[csv_col] = segyio.tracefield.keys[header_name]
                elif header_name in self._header_map:
                     mapping_resolved[csv_col] = self._header_map[header_name]
                else:
                     return False, f"Unknown SEG-Y header: {header_name}"

            # 2. Open Files
            with segyio.open(self.file_path, 'r', ignore_geometry=True) as src:
                spec = segyio.spec()
                spec.sorting = src.sorting
                spec.format = 5 # FORCE IEEE Float (Big Endian)
                spec.samples = src.samples
                spec.tracecount = src.tracecount
                
                with segyio.create(output_segy, spec) as dst:
                    # Copy Global Headers
                    dst.text[0] = src.text[0]
                    dst.bin = src.bin
                    # Force IEEE Float
                    dst.bin[segyio.BinField.Format] = 5
                    
                    # 3. Stream Traces
                    for i in range(self.n_traces):
                        # 1. Copy original header to the new file FIRST
                        dst.header[i] = src.header[i]
                        
                        # 2. Apply Patch to the DESTINATION file (which is writable)
                        row = df.iloc[i]
                        for csv_col, field_key in mapping_resolved.items():
                             val = row[csv_col]
                             # Type safety: headers are ints
                             dst.header[i][field_key] = int(val)
                        
                        # 3. Copy Data (segyio handles format conversion if needed)
                        dst.trace[i] = src.trace[i]
                        
                        if progress_callback and i % 1000 == 0:
                             progress_callback(int((i / self.n_traces) * 100))
                             
            return True, "Patching completed successfully (Standardized to Big Endian)."
            
        except Exception as e_segy:
            # If standard segyio failed (or we are in fallback mode), try raw binary patching
            if self._use_fallback or "segy" in str(e_segy).lower():
                print(f"Standard patch failed ({e_segy}), switching to RAW binary patch...")
                return self._patch_headers_raw(input_csv, output_segy, mapping, progress_callback)
            
            import traceback
            traceback.print_exc()
            return False, f"Patching failed using segyio: {str(e_segy)}"

    def _patch_headers_raw(self, input_csv, output_segy, mapping, progress_callback=None):
        """
        Manually copies file byte-by-byte and updates headers using struct packing.
        Used when segyio cannot handle the file (Fallback mode).
        Enforces Big Endian output.
        """
        try:
            import pandas as pd
            import struct
            
            # 1. Read CSV
            df = pd.read_csv(input_csv)
            n_rows = len(df)
            if n_rows != self.n_traces: return False, f"Mismatch: CSV {n_rows} != SegY {self.n_traces}"

            # 2. Resolve Mapping to (Offset, Format)
            # We only support standard fields in fallback mode
            patch_ops = [] # List of (offset, format, col_name)
            
            for csv_col, header_name in mapping.items():
                if header_name not in self._header_map:
                    print(f"Warning: Raw patch skipping '{header_name}' (not in simple map)")
                    continue
                    
                offset = self._header_map[header_name]
                # Determine type
                is_short = header_name in ['SourceGroupScalar', 'CoordinateUnits', 'TraceIdentificationCode']
                # FORCE Big Endian format for patching
                fmt = f'>h' if is_short else f'>i'
                patch_ops.append((offset, fmt, csv_col))
                
            if not patch_ops: return False, "No valid headers to patch in fallback mode."

            # 3. Raw Stream Copy with Conversion
            with open(self.file_path, 'rb') as src, open(output_segy, 'wb') as dst:
                # Read Text + Binary (3600)
                # Need to verify if we need to swap Binary Header
                header_block = bytearray(src.read(3600))
                
                # Fix Binary Header (Force IEEE, Swap if needed)
                current_bin = self.get_binary_header()
                current_bin['Data Sample Format Code'] = 5
                
                # Repack known fields as Big Endian
                for name, val in current_bin.items():
                    if name in self.BINARY_FIELD_MAP:
                        _, offset, fmt = self.BINARY_FIELD_MAP[name]
                        abs_offset = 3200 + offset
                        pack_fmt = f">{fmt}"
                        packed_val = struct.pack(pack_fmt, int(val))
                        for b_idx, b in enumerate(packed_val):
                            header_block[abs_offset + b_idx] = b

                dst.write(header_block)
                
                trace_data_len = self.n_samples * 4 # Assuming float32/int32
                trace_block_size = 240 + trace_data_len
                
                for i in range(self.n_traces):
                    # Read Trace Header (240 bytes)
                    block = src.read(trace_block_size)
                    if len(block) < trace_block_size: break
                    
                    trace_head = bytearray(block[:240])
                    trace_data = block[240:]
                    
                    # A. Swap existing Trace Header fields if input is Little Endian
                    if self._endian == '<':
                         for h_name, h_off in self._header_map.items():
                                is_short = h_name in ['SourceGroupScalar', 'CoordinateUnits', 'TraceIdentificationCode', 
                                                    'SamplesPerTrace', 'SampleInterval', 'TraceSortingCode', 
                                                    'DataUse', 'DayOfYear', 'HourOfDay', 'MinuteOfHour', 'SecondOfMinute', 'TimeBasisCode', 'WeightingFactor', 'GeophoneGroupNumberRoll1', 'GeophoneGroupTraceNumber1', 'GeophoneGroupNumberLastTrace', 'GeophoneGroupTraceNumberLast', 'GapSize', 'TaperOvertravel', 'InstrumentationSerial', 'InstrumentationSerial2', 'InstrumentationSerial3', 'ShotPoint'] 

                                fmt_char = 'h' if is_short else 'i'
                                if h_name in ['SamplesPerTrace', 'SampleInterval', 'ns', 'dt']: fmt_char = 'H'
                                else: fmt_char = 'i'
                                if h_name in ['coordinate_units', 'CoordinateUnits', 'SourceGroupScalar', 'ElevationScalar']: fmt_char = 'h'
                                
                                try:
                                    val = struct.unpack_from(f"{self._endian}{fmt_char}", trace_head, h_off)[0]
                                    struct.pack_into(f">{fmt_char}", trace_head, h_off, val)
                                except Exception as e:
                                    print(f"SeisPlotPy Warning [Patch Header Endian Swap]: {e}")
                    
                    # B. Apply CSV Patches (Already configured as Big Endian >)
                    row = df.iloc[i]
                    for offset, fmt, col_name in patch_ops:
                        val = int(row[col_name])
                        struct.pack_into(fmt, trace_head, offset, val)
                        
                    # C. Convert Data Samples
                    data_arr = np.frombuffer(trace_data, dtype=f'{self._endian}f4')
                    data_be = data_arr.astype('>f4')
                    
                    dst.write(trace_head)
                    dst.write(data_be.tobytes())
                    
                    if progress_callback and i % 1000 == 0:
                        progress_callback(int((i / self.n_traces) * 100))
                        
            return True, "Raw patching completed successfully (Standardized to Big Endian)."
            
        except Exception as e:
            return False, f"Raw patching failed: {str(e)}"