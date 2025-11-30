from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, 
                             QDialogButtonBox, QComboBox, QRadioButton, 
                             QButtonGroup, QHBoxLayout)

class BandpassDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bandpass Filter")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Low Cut (Hz):")); self.spin_low = QDoubleSpinBox(); self.spin_low.setRange(1, 200); self.spin_low.setValue(8); layout.addWidget(self.spin_low)
        layout.addWidget(QLabel("High Cut (Hz):")); self.spin_high = QDoubleSpinBox(); self.spin_high.setRange(5, 500); self.spin_high.setValue(60); layout.addWidget(self.spin_high)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
    def get_values(self): return self.spin_low.value(), self.spin_high.value()

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