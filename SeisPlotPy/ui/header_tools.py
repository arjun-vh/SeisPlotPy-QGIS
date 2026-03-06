from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QCheckBox, QDialogButtonBox, 
                             QLabel, QTextEdit, QComboBox, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QWidget, QSpinBox,
                             QGroupBox, QLineEdit, QFileDialog, QMessageBox)
import pyqtgraph as pg
from qgis.PyQt.QtCore import Qt
import numpy as np
import pandas as pd
import os

class TextHeaderDialog(QDialog):
    def __init__(self, text_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SEG-Y Text Header (EBCDIC/ASCII)")
        self.resize(600, 700)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text_content)
        self.text_edit.setReadOnly(False) # Allow editing
        self.text_edit.setStyleSheet("font-family: Courier New; font-size: 10pt;")
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        btn_save_as = QPushButton("Save As New File...")
        btn_save_as.setToolTip("Export a new SEG-Y file with this modified text header")
        btn_save_as.clicked.connect(self.accept_save)
        btn_layout.addWidget(btn_save_as)
        
        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        self.modified_text = None

    def accept_save(self):
        self.modified_text = self.text_edit.toPlainText()
        self.accept()

class HeaderQCPlot(QDialog):
    def __init__(self, available_headers, data_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trace Header QC Plot")
        self.resize(900, 600)
        self.data_manager = data_manager
        layout = QVBoxLayout(self)
        ctrl_layout = QHBoxLayout(); ctrl_layout.addWidget(QLabel("Select Header to QC:"))
        self.combo_headers = QComboBox(); self.combo_headers.addItems(available_headers); ctrl_layout.addWidget(self.combo_headers)
        self.btn_plot = QPushButton("Plot"); ctrl_layout.addWidget(self.btn_plot); layout.addLayout(ctrl_layout)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setBackground('w'); self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', "Trace Index"); self.plot_widget.setLabel('left', "Header Value"); layout.addWidget(self.plot_widget)
        self.btn_plot.clicked.connect(self.update_plot)
    def update_plot(self):
        header = self.combo_headers.currentText()
        try:
            y_vals = self.data_manager.get_header_slice(header, 0, self.data_manager.n_traces, step=1)
            x_vals = np.arange(len(y_vals)); self.plot_widget.clear()
            scatter = pg.ScatterPlotItem(x=x_vals, y=y_vals, pen=None, symbol='o', size=3, brush=pg.mkBrush(0, 0, 255, 100))
            self.plot_widget.addItem(scatter)
        except Exception as e: print(f"QC Plot error: {e}")

class SpectrumPlot(QDialog):
    def __init__(self, freqs, amps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Average Frequency Spectrum")
        self.resize(800, 500)
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setBackground('w'); self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', "Frequency", units='Hz'); self.plot_widget.setLabel('left', "Average Amplitude")
        self.plot_widget.plot(freqs, amps, pen='b', fillLevel=0, brush=(0, 0, 255, 50)); layout.addWidget(self.plot_widget)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.accept); layout.addWidget(btn_close)

class HeaderExportDialog(QDialog):
    def __init__(self, available_headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Trace Headers to CSV")
        self.resize(400, 600)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select trace headers to export:"))
        
        self.list_widget = QListWidget()
        for h in available_headers:
            item = QListWidgetItem(h)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        
        # Selection Buttons
        btn_box = QHBoxLayout()
        btn_all = QPushButton("Select All"); btn_all.clicked.connect(self.sel_all)
        btn_none = QPushButton("Select None"); btn_none.clicked.connect(self.sel_none)
        btn_box.addWidget(btn_all); btn_box.addWidget(btn_none)
        layout.addLayout(btn_box)
        
        # Include Index Checkbox
        self.chk_index = QCheckBox("Include Trace Index (Recommended)")
        self.chk_index.setChecked(True)
        self.chk_index.setToolTip("Adds a sequential 0-based index column to help with matching")
        layout.addWidget(self.chk_index)
        
        # Standard Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def sel_all(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.Checked)
    
    def sel_none(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.Unchecked)
    
    def get_selected_headers(self):
        selected = []
        if self.chk_index.isChecked():
            selected.append("TraceIndex")
            
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

class HeaderPatchDialog(QDialog):
    def __init__(self, available_headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Patch Headers from CSV")
        self.resize(600, 500)
        self.available_headers = available_headers
        self.csv_columns = []
        
        layout = QVBoxLayout(self)
        
        # 1. Input CSV
        input_group = QGroupBox("1. Input Data")
        input_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.txt_csv = QLineEdit(); self.txt_csv.setPlaceholderText("Select CSV file...")
        self.btn_browse_csv = QPushButton("Browse..."); self.btn_browse_csv.clicked.connect(self.browse_csv)
        file_layout.addWidget(self.txt_csv); file_layout.addWidget(self.btn_browse_csv)
        input_layout.addLayout(file_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 2. Mapping Table
        map_group = QGroupBox("2. Header Mapping")
        map_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["CSV Column", "Target SEG-Y Header"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        map_layout.addWidget(self.table)
        
        map_group.setLayout(map_layout)
        layout.addWidget(map_group)
        
        # 3. Output
        out_group = QGroupBox("3. Output")
        out_layout = QHBoxLayout()
        self.txt_out = QLineEdit(); self.txt_out.setPlaceholderText("Output SEG-Y path...")
        self.btn_browse_out = QPushButton("Browse..."); self.btn_browse_out.clicked.connect(self.browse_out)
        out_layout.addWidget(self.txt_out); out_layout.addWidget(self.btn_browse_out)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv *.txt)")
        if path:
            self.txt_csv.setText(path)
            self.load_csv_headers(path)
            
    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save New SEG-Y", "", "SEG-Y Files (*.sgy *.segy)")
        if path:
            if not path.lower().endswith(('.sgy', '.segy')): path += '.sgy'
            self.txt_out.setText(path)

    def load_csv_headers(self, path):
        try:
            # Read just the first line to get columns
            import pandas as pd
            df = pd.read_csv(path, nrows=0)
            self.csv_columns = list(df.columns)
            
            self.table.setRowCount(len(self.csv_columns))
            
            # Populate table
            for i, col in enumerate(self.csv_columns):
                self.table.setItem(i, 0, QTableWidgetItem(col))
                
                # Combo box for mapping
                combo = QComboBox()
                combo.addItem("-- Ignore --")
                combo.addItems(self.available_headers)
                
                # Auto-match logic
                normalized_col = col.lower().strip()
                match = next((h for h in self.available_headers if h.lower() == normalized_col), None)
                if match:
                    combo.setCurrentText(match)
                
                self.table.setCellWidget(i, 1, combo)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV: {e}")

    def validate(self):
        if not self.txt_csv.text() or not self.txt_out.text():
             QMessageBox.warning(self, "Missing Files", "Please select input CSV and output SEG-Y paths.")
             return
             
        # Check if at least one column is mapped
        mapping = self.get_mapping()
        if not mapping:
            QMessageBox.warning(self, "No Mapping", "Please map at least one CSV column to a SEG-Y header.")
            return
            
        self.accept()
        
    def get_inputs(self):
        return self.txt_csv.text(), self.txt_out.text(), self.get_mapping()
        
    def get_mapping(self):
        # Returns dict: {csv_col_name: segy_header_key}
        mapping = {}
        for i in range(self.table.rowCount()):
            combo = self.table.cellWidget(i, 1)
            target = combo.currentText()
            if target != "-- Ignore --":
                csv_col = self.table.item(i, 0).text()
                mapping[csv_col] = target
        return mapping
    
class HeaderExplorer(QDialog):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setWindowTitle(f"Header Explorer: {os.path.basename(data_manager.file_path)}")
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # 1. Binary Header Tab
        # 1. Binary Header Tab
        bin_container = QWidget()
        bin_layout = QVBoxLayout(bin_container)

        self.bin_table = QTableWidget()
        self.bin_table.setColumnCount(3)
        self.bin_table.setHorizontalHeaderLabels(["Field Name", "Value", "Detected (Ref)"])
        self.bin_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bin_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        bin_layout.addWidget(self.bin_table)
        
        self.btn_save_bin = QPushButton("Save Binary Header Changes As...")
        self.btn_save_bin.setToolTip("Export a new SEG-Y file with updated binary header values")
        self.btn_save_bin.setStyleSheet("font-weight: bold; background-color: #e6f3ff; Padding: 5px;")
        self.btn_save_bin.clicked.connect(self.save_binary_changes)
        bin_layout.addWidget(self.btn_save_bin)
        
        self.tabs.addTab(bin_container, "Binary File Header")
        
        # 2. Trace Header Tab Container
        trace_container = QWidget()
        trace_layout = QVBoxLayout(trace_container)
        
        self.trace_table = QTableWidget()
        trace_layout.addWidget(self.trace_table)
        
        # --- Range Selection UI ---
        ctrl_layout = QHBoxLayout()
        max_idx = self.data_manager.n_traces - 1
        
        # Display the file limits for user reference
        lbl_limits = QLabel(f"(Available Range: 0 to {max_idx})")
        lbl_limits.setStyleSheet("color: blue; font-weight: bold; margin-right: 10px;")

        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, max_idx)
        self.spin_start.setValue(0) # Default start

        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, max_idx)
        self.spin_end.setValue(min(1, max_idx)) # Default end (shows index 0 and 1)

        self.btn_refresh = QPushButton("Refresh Range")
        self.btn_refresh.clicked.connect(self.populate_trace_headers)

        ctrl_layout.addWidget(lbl_limits)
        ctrl_layout.addWidget(QLabel("From Index:"))
        ctrl_layout.addWidget(self.spin_start)
        ctrl_layout.addWidget(QLabel("To Index:"))
        ctrl_layout.addWidget(self.spin_end)
        ctrl_layout.addWidget(self.btn_refresh)
        ctrl_layout.addStretch()
        
        trace_layout.addLayout(ctrl_layout)
        
        self.tabs.addTab(trace_container, "Trace Headers")
        layout.addWidget(self.tabs)
        
        self.populate_binary_header()
        self.populate_trace_headers()

    def populate_binary_header(self):
        # We fetch the dictionary from our data_manager
        bin_data = self.data_manager.get_binary_header()
        
        if not bin_data:
            self.bin_table.setRowCount(1)
            self.bin_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.bin_table.setItem(0, 1, QTableWidgetItem("Could not read binary block"))
            return

        self.bin_table.setRowCount(len(bin_data))
        for i, (name, val) in enumerate(bin_data.items()):
            # Name Item (Read Only)
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.bin_table.setItem(i, 0, item_name)
            
            # Value Item (Editable)
            item_val = QTableWidgetItem(str(val))
            item_val.setFlags(item_val.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled)
            item_val.setBackground(pg.mkColor("#f0fff0")) # Light green hint
            item_val.setToolTip("Double-click to edit. Save changes with button below.")
            self.bin_table.setItem(i, 1, item_val)

            # Detected Value Item (Read Only - Reference)
            det_val = ""
            if "Samples per Trace" in name:
                det_val = str(self.data_manager.n_samples)
            elif "Sample Interval" in name and "Original" not in name:
                # Convert seconds to microseconds for comparison
                det_val = str(int(self.data_manager.sample_rate * 1000))
            
            item_det = QTableWidgetItem(det_val)
            item_det.setFlags(item_det.flags() & ~Qt.ItemIsEditable)
            
            # Highlight mismatch
            if det_val and det_val != str(val):
                item_det.setForeground(pg.mkBrush('r'))
                item_det.setToolTip(f"Mismatch! File is actually read as format {det_val} (1=IBM, 5=IEEE)")
            
            self.bin_table.setItem(i, 2, item_det)
    
    def save_binary_changes(self):
        # 1. Gather Changes
        new_values = {}
        row_count = self.bin_table.rowCount()
        
        for i in range(row_count):
            name = self.bin_table.item(i, 0).text()
            val_str = self.bin_table.item(i, 1).text()
            
            try:
                val_int = int(val_str)
                new_values[name] = val_int
            except ValueError:
                QMessageBox.warning(self, "Invalid Value", f"Value for '{name}' must be an integer.\nFound: '{val_str}'")
                return

        # 2. Get Output Path
        path, _ = QFileDialog.getSaveFileName(self, "Save New SEG-Y", "", "SEG-Y Files (*.sgy *.segy)")
        if not path:
            return
            
        if not path.lower().endswith(('.sgy', '.segy')):
             path += '.sgy'

        # 3. Execute Save
        # Add a progress dialog if needed, but for now we block the UI (simpler)
        # or use a busy cursor
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            success, msg = self.data_manager.save_new_segy_with_header(path, new_values)
            QApplication.restoreOverrideCursor()
            
            if success:
                QMessageBox.information(self, "Success", f"File saved successfully:\n{path}")
            else:
                QMessageBox.critical(self, "Error", msg)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", str(e))
            
    def populate_trace_headers(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        
        # Safety check: ensure range is valid and not massive
        if end < start:
            end = start
        if (end - start) > 500:
            # Prevent UI hang if user asks for 1 million rows
            msg = "Large range selected. Loading more than 500 traces may be slow. Continue?"
            # Add QMessageBox here if desired
        
        n_to_show = (end - start) + 1
        headers = self.data_manager.available_headers
        
        self.trace_table.setRowCount(n_to_show)
        self.trace_table.setColumnCount(len(headers))
        self.trace_table.setHorizontalHeaderLabels(headers)
        
        for col_idx, h_name in enumerate(headers):
            # Fetch exactly the requested range
            vals = self.data_manager.get_header_slice(h_name, start, end + 1, 1)
            for row_idx, v in enumerate(vals):
                self.trace_table.setVerticalHeaderItem(row_idx, QTableWidgetItem(str(start + row_idx)))
                self.trace_table.setItem(row_idx, col_idx, QTableWidgetItem(str(v)))