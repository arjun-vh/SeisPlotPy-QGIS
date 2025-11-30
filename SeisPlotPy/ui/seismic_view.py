from qgis.PyQt.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QSplitter, 
                             QComboBox, QDoubleSpinBox, QCheckBox, QGroupBox)
from qgis.PyQt.QtCore import Qt, QRectF
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SeismicView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeisPlotPy v2")
        self.resize(1200, 800)
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        self.layout = QHBoxLayout(main_widget); self.layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(280); self.sidebar.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ccc;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.btn_load = QPushButton("Load SEG-Y"); self.btn_load.setMinimumHeight(40); self.sidebar_layout.addWidget(self.btn_load)
        self.sidebar_layout.addSpacing(5); data_group = QGroupBox("Active Viewport"); data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel("X-Axis Reference:")); self.combo_header = QComboBox(); self.combo_header.addItem("Trace Index"); data_layout.addWidget(self.combo_header)
        data_layout.addWidget(QLabel("X-Axis Range:")); x_layout = QHBoxLayout(); self.spin_x_min = QDoubleSpinBox(); self.spin_x_min.setRange(-99999999, 99999999); self.spin_x_min.setDecimals(2); x_layout.addWidget(self.spin_x_min); self.spin_x_max = QDoubleSpinBox(); self.spin_x_max.setRange(-99999999, 99999999); self.spin_x_max.setDecimals(2); x_layout.addWidget(self.spin_x_max); data_layout.addLayout(x_layout)
        data_layout.addWidget(QLabel("Y-Axis Range:")); y_layout = QHBoxLayout(); self.spin_y_min = QDoubleSpinBox(); self.spin_y_min.setRange(-10000, 50000); y_layout.addWidget(self.spin_y_min); self.spin_y_max = QDoubleSpinBox(); self.spin_y_max.setRange(-10000, 50000); y_layout.addWidget(self.spin_y_max); data_layout.addLayout(y_layout)
        data_layout.addWidget(QLabel("Decimation (Step):")); step_layout = QHBoxLayout(); self.chk_manual_step = QCheckBox("Manual"); self.chk_manual_step.setToolTip("Check to force a specific step size"); step_layout.addWidget(self.chk_manual_step); self.spin_step = QDoubleSpinBox(); self.spin_step.setDecimals(0); self.spin_step.setRange(1, 5000); self.spin_step.setValue(1); self.spin_step.setEnabled(False); step_layout.addWidget(self.spin_step); data_layout.addLayout(step_layout)
        btn_box = QHBoxLayout(); self.btn_apply = QPushButton("Apply / Reload"); self.btn_apply.setStyleSheet("background-color: #d0e0ff; font-weight: bold;"); self.btn_apply.setToolTip("Reloads data for the selected range"); btn_box.addWidget(self.btn_apply); self.btn_reset = QPushButton("Reset View"); self.btn_reset.setToolTip("Zoom out to full extent"); btn_box.addWidget(self.btn_reset); data_layout.addLayout(btn_box); data_group.setLayout(data_layout); self.sidebar_layout.addWidget(data_group)
        self.sidebar_layout.addWidget(QLabel("<b>Visualization</b>")); self.sidebar_layout.addWidget(QLabel("Domain:")); self.combo_domain = QComboBox(); self.combo_domain.addItems(["Time", "Depth"]); self.sidebar_layout.addWidget(self.combo_domain)
        toggle_layout = QHBoxLayout(); self.chk_flip_x = QCheckBox("Flip X"); self.chk_grid = QCheckBox("Grid"); self.chk_grid.setChecked(True); toggle_layout.addWidget(self.chk_flip_x); toggle_layout.addWidget(self.chk_grid); self.sidebar_layout.addLayout(toggle_layout)
        self.sidebar_layout.addWidget(QLabel("Colormap:")); self.combo_cmap = QComboBox(); all_cmaps = sorted(plt.colormaps()); self.combo_cmap.addItems(all_cmaps); self.combo_cmap.setCurrentText("seismic"); self.sidebar_layout.addWidget(self.combo_cmap)
        self.sidebar_layout.addWidget(QLabel("Contrast (Percentile):")); self.spin_contrast = QDoubleSpinBox(); self.spin_contrast.setRange(50.0, 100.0); self.spin_contrast.setValue(99.0); self.spin_contrast.setSingleStep(0.1); self.sidebar_layout.addWidget(self.spin_contrast); self.sidebar_layout.addSpacing(10)
        export_group = QGroupBox("Export"); export_layout = QVBoxLayout(); dim_layout = QHBoxLayout(); dim_layout.addWidget(QLabel("W (in):")); self.spin_fig_width = QDoubleSpinBox(); self.spin_fig_width.setValue(10.0); dim_layout.addWidget(self.spin_fig_width); dim_layout.addWidget(QLabel("H (in):")); self.spin_fig_height = QDoubleSpinBox(); self.spin_fig_height.setValue(6.0); dim_layout.addWidget(self.spin_fig_height); export_layout.addLayout(dim_layout)
        self.btn_preview_ratio = QPushButton("Match Aspect Ratio"); export_layout.addWidget(self.btn_preview_ratio); self.btn_export = QPushButton("Export Figure"); self.btn_export.setStyleSheet("background-color: #ffcccc; font-weight: bold;"); export_layout.addWidget(self.btn_export); export_group.setLayout(export_layout); self.sidebar_layout.addWidget(export_group)
        self.sidebar_layout.addStretch(); self.lbl_info = QLabel("No file loaded"); self.lbl_info.setWordWrap(True); self.sidebar_layout.addWidget(self.lbl_info)
        
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setBackground('w'); self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.img_item = pg.ImageItem(); self.plot_widget.addItem(self.img_item); self.plot_widget.getPlotItem().invertY(True)
        
        
        self.plot_widget.setLabel('left', 'Time (ms)')
        self.plot_widget.setLabel('bottom', 'Trace Index')
        
        
        self.plot_widget.getPlotItem().setAspectLocked(False) 
        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(self.sidebar); splitter.addWidget(self.plot_widget); self.layout.addWidget(splitter)

    def update_status(self, message): self.lbl_info.setText(message)
    def set_colormap(self, name):
        try: colormap = cm.get_cmap(name)
        except: colormap = cm.get_cmap("gray")
        lut = (colormap(np.arange(256)) * 255).astype(np.uint8); self.img_item.setLookupTable(lut)
    
    
    def update_labels(self, x_label, y_domain):
        self.plot_widget.setLabel('bottom', x_label)
        if y_domain == "Time": 
            self.plot_widget.setLabel('left', 'TWT (ms)') 
        else: 
            self.plot_widget.setLabel('left', 'Depth (m)')
    

    def display_seismic(self, data_array, x_range=None, y_range=None):
        self.img_item.setImage(data_array, autoLevels=False)
        if x_range is not None and y_range is not None:
            x_min, x_max = x_range; y_min, y_max = y_range; width = x_max - x_min; height = y_max - y_min; self.img_item.setRect(QRectF(x_min, y_min, width, height))
        self.set_colormap(self.combo_cmap.currentText())