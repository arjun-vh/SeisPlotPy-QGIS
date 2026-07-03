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
        self.setWindowTitle("SeisPlotPy")
        self.resize(1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QHBoxLayout(main_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(280)
        self.sidebar.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ccc;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        
        load_layout = QHBoxLayout()
        self.btn_load_single = QPushButton("Load Single\nSEG-Y")
        self.btn_load_single.setMinimumHeight(45)
        
        self.btn_load_batch = QPushButton("Batch load\nmultiple SEG-Y")
        self.btn_load_batch.setMinimumHeight(45)
        
        load_layout.addWidget(self.btn_load_single)
        load_layout.addWidget(self.btn_load_batch)
        self.sidebar_layout.addLayout(load_layout)
        
        self.sidebar_layout.addSpacing(5)
        
        # Data Group
        data_group = QGroupBox("Active Viewport")
        data_layout = QVBoxLayout()
        
        data_layout.addWidget(QLabel("X-Axis Reference:"))
        self.combo_header = QComboBox()
        self.combo_header.addItem("Trace Index")
        data_layout.addWidget(self.combo_header)
        
        data_layout.addWidget(QLabel("X-Axis Range:"))
        x_layout = QHBoxLayout()
        self.spin_x_min = QDoubleSpinBox(); self.spin_x_min.setRange(-99999999, 99999999); self.spin_x_min.setDecimals(2)
        x_layout.addWidget(self.spin_x_min)
        self.spin_x_max = QDoubleSpinBox(); self.spin_x_max.setRange(-99999999, 99999999); self.spin_x_max.setDecimals(2)
        x_layout.addWidget(self.spin_x_max)
        data_layout.addLayout(x_layout)
        
        data_layout.addWidget(QLabel("Y-Axis Range:"))
        y_layout = QHBoxLayout()
        self.spin_y_min = QDoubleSpinBox(); self.spin_y_min.setRange(-10000, 50000)
        y_layout.addWidget(self.spin_y_min)
        self.spin_y_max = QDoubleSpinBox(); self.spin_y_max.setRange(-10000, 50000)
        y_layout.addWidget(self.spin_y_max)
        data_layout.addLayout(y_layout)
        
        data_layout.addWidget(QLabel("Decimation (Step):"))
        step_layout = QHBoxLayout()
        self.chk_manual_step = QCheckBox("Manual")
        self.chk_manual_step.setToolTip("Check to force a specific step size")
        step_layout.addWidget(self.chk_manual_step)
        self.spin_step = QDoubleSpinBox(); self.spin_step.setDecimals(0); self.spin_step.setRange(1, 5000); self.spin_step.setValue(1); self.spin_step.setEnabled(False)
        step_layout.addWidget(self.spin_step)
        data_layout.addLayout(step_layout)
        
        btn_box = QHBoxLayout()
        self.btn_apply = QPushButton("Apply / Reload")
        self.btn_apply.setStyleSheet("background-color: #d0e0ff; font-weight: bold;")
        self.btn_apply.setToolTip("Reloads data for the selected range")
        btn_box.addWidget(self.btn_apply)
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.setToolTip("Zoom out to full extent")
        btn_box.addWidget(self.btn_reset)
        data_layout.addLayout(btn_box)
        
        data_group.setLayout(data_layout)
        self.sidebar_layout.addWidget(data_group)
        
        # Visualization
        self.sidebar_layout.addWidget(QLabel("<b>Visualization</b>"))
        self.sidebar_layout.addWidget(QLabel("Domain:"))
        self.combo_domain = QComboBox()
        self.combo_domain.addItems(["Time", "Depth"])
        self.sidebar_layout.addWidget(self.combo_domain)
        
        # Toggle Layout (Flip, Grid, Smooth)
        # Toggle Layout (Flip, Grid, Smooth, High Res)
        from qgis.PyQt.QtWidgets import QGridLayout
        toggle_layout = QGridLayout()
        
        self.chk_flip_x = QCheckBox("Flip X")
        self.chk_grid = QCheckBox("Grid"); self.chk_grid.setChecked(True)
        self.btn_grid_cfg = QPushButton("⚙")
        self.btn_grid_cfg.setFixedWidth(20)
        self.btn_grid_cfg.setToolTip("Configure Grid settings")
        
        grid_layout = QHBoxLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(self.chk_grid)
        grid_layout.addWidget(self.btn_grid_cfg)
        grid_layout.addStretch()
        
        self.chk_smooth = QCheckBox("Smooth")
        self.chk_smooth.setToolTip("Enable bilinear interpolation (anti-aliasing)")

        self.chk_high_res = QCheckBox("High Res")
        self.chk_high_res.setToolTip("Interpolate data (4x) for vector-like display. (CPU Intensive)")
        
        self.btn_high_res_cfg = QPushButton("⚙")
        self.btn_high_res_cfg.setFixedWidth(20)
        self.btn_high_res_cfg.setToolTip("Configure High-Res mode settings")
        
        high_res_layout = QHBoxLayout()
        high_res_layout.setContentsMargins(0, 0, 0, 0)
        high_res_layout.addWidget(self.chk_high_res)
        high_res_layout.addWidget(self.btn_high_res_cfg)
        high_res_layout.addStretch()

        # Row 0
        toggle_layout.addWidget(self.chk_flip_x, 0, 0)
        toggle_layout.addLayout(grid_layout, 0, 1)
        
        # Row 1
        toggle_layout.addWidget(self.chk_smooth, 1, 0)
        toggle_layout.addLayout(high_res_layout, 1, 1)

        self.sidebar_layout.addLayout(toggle_layout)
        
        self.sidebar_layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        from qgis.PyQt.QtCore import QSize
        self.combo_cmap.setIconSize(QSize(80, 14))
        
        try:
            all_cmaps = sorted(plt.colormaps())
        except Exception:
            # Fallback to common colormap names if API fails
            all_cmaps = ['viridis', 'seismic', 'gray', 'jet', 'RdBu', 'hot', 'cool']
            
        from qgis.PyQt.QtGui import QImage, QPixmap, QIcon, QColor
        for cmap_name in all_cmaps:
            try:
                if hasattr(plt, 'colormaps') and hasattr(plt.colormaps, '__getitem__'):
                    cmap = plt.colormaps[cmap_name]
                else:
                    cmap = cm.get_cmap(cmap_name)
                    
                width = 80
                gradient = np.linspace(0, 1, width)
                rgba = cmap(gradient)
                
                qimg = QImage(width, 1, QImage.Format_ARGB32)
                for x in range(width):
                    r, g, b, a = rgba[x]
                    qimg.setPixelColor(x, 0, QColor(int(r*255), int(g*255), int(b*255), int(a*255)))
                
                pixmap = QPixmap.fromImage(qimg.scaled(width, 14))
                icon = QIcon(pixmap)
                self.combo_cmap.addItem(icon, cmap_name)
            except Exception:
                self.combo_cmap.addItem(cmap_name)
                
        self.combo_cmap.setCurrentText("seismic")
        self.sidebar_layout.addWidget(self.combo_cmap)
        
        self.sidebar_layout.addWidget(QLabel("Contrast (Percentile):"))
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(50.0, 100.0); self.spin_contrast.setValue(99.0); self.spin_contrast.setSingleStep(0.1)
        self.sidebar_layout.addWidget(self.spin_contrast)
        
        self.chk_show_legend = QCheckBox("Show Color Legend")
        self.chk_show_legend.setChecked(False)
        self.sidebar_layout.addWidget(self.chk_show_legend)
        
        self.sidebar_layout.addSpacing(10)
        
        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("W (in):"))
        self.spin_fig_width = QDoubleSpinBox(); self.spin_fig_width.setValue(10.0)
        dim_layout.addWidget(self.spin_fig_width)
        dim_layout.addWidget(QLabel("H (in):"))
        self.spin_fig_height = QDoubleSpinBox(); self.spin_fig_height.setValue(6.0)
        dim_layout.addWidget(self.spin_fig_height)
        export_layout.addLayout(dim_layout)
        
        self.btn_preview_ratio = QPushButton("Match Aspect Ratio")
        export_layout.addWidget(self.btn_preview_ratio)
        self.btn_export = QPushButton("Export Figure")
        self.btn_export.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        export_layout.addWidget(self.btn_export)
        export_group.setLayout(export_layout)
        self.sidebar_layout.addWidget(export_group)
        
        self.sidebar_layout.addStretch()
        self.lbl_info = QLabel("No file loaded")
        self.lbl_info.setWordWrap(True)
        self.sidebar_layout.addWidget(self.lbl_info)
        
        # --- PLOT AREA ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.plot_widget.getPlotItem().invertY(True)
        
        self.plot_widget.setLabel('left', 'Time (ms)')
        self.plot_widget.setLabel('bottom', 'Trace Index')
        self.plot_widget.getPlotItem().setAspectLocked(False) 
        
        # --- COLORBAR WIDGET ---
        self.color_bar = pg.HistogramLUTWidget(image=self.img_item)
        self.color_bar.setFixedWidth(120)
        self.color_bar.hide() # Hidden by default
        
        # Enforce Single Source of Truth: Disable the internal context menu 
        # so the user must use the main Matplotlib combo box for colormaps.
        self.color_bar.vb.setMenuEnabled(False)
        if hasattr(self.color_bar.gradient, 'menu'):
            self.color_bar.gradient.menu = None
            
        # Override the right click event to prevent the menu from popping up
        def null_mouse_click(ev):
            if ev.button() == Qt.RightButton:
                ev.accept()
            else:
                # Call original for left clicks (to allow moving the gradient ticks if desired)
                pg.TickSliderItem.mouseClickEvent(self.color_bar.gradient, ev)
                
        self.color_bar.gradient.mouseClickEvent = null_mouse_click
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.sidebar)
        
        # Container for plot and colorbar
        plot_container = QWidget()
        plot_layout = QHBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addWidget(self.color_bar)
        
        splitter.addWidget(plot_container)
        self.layout.addWidget(splitter)

        # --- NEW CODE: Enable Status Bar for Coordinates ---
        self.status_bar = self.statusBar()
        self.lbl_coords = QLabel("X: - | Y: -")
        self.lbl_coords.setStyleSheet("font-weight: bold; margin-right: 15px;")
        self.status_bar.addPermanentWidget(self.lbl_coords) # Adds it to the far right

    def update_status(self, message): self.lbl_info.setText(message)
    
    def set_colormap(self, name):
        try:
            # Modern Matplotlib (3.7+)
            if hasattr(plt, 'colormaps') and hasattr(plt.colormaps, '__getitem__'):
                 colormap = plt.colormaps[name]
            else:
                 # Legacy
                 colormap = cm.get_cmap(name)
        except Exception:
             colormap = cm.get_cmap("gray")
             
        lut = (colormap(np.arange(256)) * 255).astype(np.uint8)
        self.img_item.setLookupTable(lut)
        
        try:
            # Synchronize gradient on the colorbar
            # Older pyqtgraph uses restoreState, newer uses setColorMap
            cmap_pg = pg.ColorMap(np.linspace(0.0, 1.0, 256), lut)
            self.color_bar.gradient.setColorMap(cmap_pg)
        except Exception:
            pass
    
    def update_labels(self, x_label, y_domain):
        self.plot_widget.setLabel('bottom', x_label)
        if y_domain == "Time": 
            self.plot_widget.setLabel('left', 'TWT (ms)') 
        else: 
            self.plot_widget.setLabel('left', 'Depth (m)')

    def display_seismic(self, data_array, x_range=None, y_range=None):
        self.img_item.setImage(data_array, autoLevels=False)
        if x_range is not None and y_range is not None:
            x_min, x_max = x_range; y_min, y_max = y_range
            width = x_max - x_min
            height = y_max - y_min
            self.img_item.setRect(QRectF(x_min, y_min, width, height))
        self.set_colormap(self.combo_cmap.currentText())

    def show_message(self, title, message, level="info"):
        """Displays a message box with the specified level."""
        from qgis.PyQt.QtWidgets import QMessageBox
        if level == "info":
            QMessageBox.information(self, title, message)
        elif level == "warning":
            QMessageBox.warning(self, title, message)
        elif level == "critical":
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.information(self, title, message)