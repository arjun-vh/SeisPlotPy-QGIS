from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QCheckBox, QDialogButtonBox, 
                             QLabel, QTextEdit, QComboBox)
import pyqtgraph as pg
from qgis.PyQt.QtCore import Qt
import numpy as np

class TextHeaderDialog(QDialog):
    def __init__(self, text_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SEG-Y Text Header (EBCDIC/ASCII)")
        self.resize(600, 700)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text_content)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("font-family: Courier New; font-size: 10pt;")
        layout.addWidget(self.text_edit)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.accept); layout.addWidget(btn_close)

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
        self.setWindowTitle("Export Horizon with Headers")
        self.resize(400, 500)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Select trace headers to include in CSV:"))
        self.list_widget = QListWidget()
        for h in available_headers:
            item = QListWidgetItem(h); item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        btn_box = QHBoxLayout(); btn_all = QPushButton("Select All"); btn_all.clicked.connect(self.sel_all)
        btn_none = QPushButton("Select None"); btn_none.clicked.connect(self.sel_none)
        btn_box.addWidget(btn_all); btn_box.addWidget(btn_none); layout.addLayout(btn_box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
    def sel_all(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.Checked)
    def sel_none(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.Unchecked)
    def get_selected_headers(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked: selected.append(item.text())
        return selected