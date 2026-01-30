from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QDialogButtonBox, QComboBox, QRadioButton, QSpinBox, QPushButton,
                             QButtonGroup, QLineEdit, QFileDialog, QMessageBox)
import os

class BandpassDialog(QDialog):
    def __init__(self, parent=None, nyquist=None):
        super().__init__(parent)

        self.setWindowTitle("Bandpass Filter")
        layout = QVBoxLayout(self)

        # -------------------------------
        # Determine maximum allowed frequency
        # -------------------------------
        if nyquist is not None:
            max_freq = 0.9 * nyquist
        else:
            max_freq = 20000  # safe fallback

        # -------------------------------
        # Low cut
        # -------------------------------
        layout.addWidget(QLabel("Low Cut (Hz):"))
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setRange(1, max_freq)
        self.spin_low.setValue(min(8, max_freq))
        layout.addWidget(self.spin_low)

        # -------------------------------
        # High cut
        # -------------------------------
        layout.addWidget(QLabel("High Cut (Hz):"))
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setRange(5, max_freq)
        self.spin_high.setValue(min(60, max_freq))
        layout.addWidget(self.spin_high)

        # -------------------------------
        # Buttons
        # -------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        return self.spin_low.value(), self.spin_high.value()


class GeometryDialog(QDialog):
    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Geometry / Distance")
        self.resize(400, 300)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Select Coordinate Headers to calculate Distance:"))
        layout.addWidget(QLabel("X Coordinate:")); self.combo_x = QComboBox(); self.combo_x.addItems(headers)
        if 'CDP_X' in headers: self.combo_x.setCurrentText('CDP_X')
        elif 'SourceX' in headers: self.combo_x.setCurrentText('SourceX')
        layout.addWidget(self.combo_x)
        layout.addWidget(QLabel("Y Coordinate:")); self.combo_y = QComboBox(); self.combo_y.addItems(headers)
        if 'CDP_Y' in headers: self.combo_y.setCurrentText('CDP_Y')
        elif 'SourceY' in headers: self.combo_y.setCurrentText('SourceY')
        layout.addWidget(self.combo_y)
        layout.addSpacing(10); layout.addWidget(QLabel("<b>Scaling Method:</b>"))
        self.bg_scalar = QButtonGroup(self)
        self.rb_header = QRadioButton("Use Scalar Header"); self.rb_header.setChecked(True); self.bg_scalar.addButton(self.rb_header); layout.addWidget(self.rb_header)
        self.combo_scalar = QComboBox(); self.combo_scalar.addItems(headers)
        if 'SourceGroupScalar' in headers: self.combo_scalar.setCurrentText('SourceGroupScalar')
        layout.addWidget(self.combo_scalar)
        self.rb_manual = QRadioButton("Manual Scalar (Override)"); self.bg_scalar.addButton(self.rb_manual); layout.addWidget(self.rb_manual)
        self.spin_manual = QDoubleSpinBox(); self.spin_manual.setRange(-1000000, 1000000); self.spin_manual.setValue(1.0); self.spin_manual.setDecimals(6); layout.addWidget(self.spin_manual)
        layout.addSpacing(20)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.rb_header.toggled.connect(self.toggle_inputs); self.toggle_inputs()
    def toggle_inputs(self):
        use_header = self.rb_header.isChecked(); self.combo_scalar.setEnabled(use_header); self.spin_manual.setEnabled(not use_header)
    def get_settings(self):
        return {'x_key': self.combo_x.currentText(), 'y_key': self.combo_y.currentText(), 'use_header': self.rb_header.isChecked(), 'scalar_key': self.combo_scalar.currentText(), 'manual_val': self.spin_manual.value()}

class ExportSubsetDialog(QDialog):
    def __init__(self, total_traces, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export SEG-Y Subset")
        self.resize(400, 200)
        
        # Store the max trace count so we can set limits
        self.total_traces = total_traces
        self.selected_file = None
        
        layout = QVBoxLayout(self)
        
        # 1. Range Selection
        range_layout = QHBoxLayout()
        
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, total_traces - 1)
        self.spin_start.setValue(0)
        self.spin_start.setPrefix("Start: ")
        
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, total_traces - 1)
        self.spin_end.setValue(min(100, total_traces - 1)) # Default to 100 traces
        self.spin_end.setPrefix("End: ")
        
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(self.spin_end)
        layout.addLayout(range_layout)
        
        # 2. File Selection
        file_layout = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Select output file...")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.txt_path)
        file_layout.addWidget(self.btn_browse)
        layout.addLayout(file_layout)
        
        # 3. OK / Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def browse_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save New SEG-Y File", "", "SEG-Y Files (*.sgy *.segy)"
        )
        if filename:
            # Ensure extension
            if not filename.lower().endswith(('.sgy', '.segy')):
                filename += '.sgy'
            self.txt_path.setText(filename)
            self.selected_file = filename

    def validate_and_accept(self):
        # Check if file is selected
        if not self.txt_path.text():
            QMessageBox.warning(self, "Missing File", "Please select an output file path.")
            return
            
        # Check if range is valid
        start = self.spin_start.value()
        end = self.spin_end.value()
        
        if start > end:
            QMessageBox.warning(self, "Invalid Range", "Start trace must be less than End trace.")
            return
            
        self.accept() # Close the dialog and return "Success" code

    def get_inputs(self):
        """Returns (output_path, start_trace, end_trace)"""
        return self.txt_path.text(), self.spin_start.value(), self.spin_end.value()