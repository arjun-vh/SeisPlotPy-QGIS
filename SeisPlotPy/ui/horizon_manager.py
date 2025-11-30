from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QFileDialog, QColorDialog, QCheckBox, QWidget, QMessageBox,
                             QLabel, QRadioButton, QButtonGroup)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, pyqtSignal

import pandas as pd
import numpy as np
import os

class HorizonManager(QDialog):
    picking_toggled = pyqtSignal(bool, str)
    horizon_visibility_changed = pyqtSignal()
    horizon_color_changed = pyqtSignal()
    horizon_removed = pyqtSignal()
    export_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Horizon Interpretation Manager")
        self.resize(600, 400)
        
        
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        self.horizons = []
        self.active_horizon_index = -1
        self.is_picking = False
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton("+ New Horizon"); self.btn_new.clicked.connect(self.create_horizon)
        self.btn_import = QPushButton("Import CSV"); self.btn_import.clicked.connect(self.import_horizon)
        btn_layout.addWidget(self.btn_new); btn_layout.addWidget(self.btn_import)
        layout.addLayout(btn_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Active", "Name", "Color", "Points", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.pick_group = QButtonGroup(self)
        
        self.lbl_status = QLabel("Status: Viewing Mode")
        self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")
        layout.addWidget(self.lbl_status)
        
        # Actions
        action_layout = QHBoxLayout()
        self.btn_pick = QPushButton("Start Picking")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setStyleSheet("background-color: #e0e0e0;")
        self.btn_pick.clicked.connect(self.toggle_picking)
        self.btn_pick.setEnabled(False)
        action_layout.addWidget(self.btn_pick)
        
        self.btn_save = QPushButton("Save Selected to CSV")
        self.btn_save.clicked.connect(self.request_export)
        action_layout.addWidget(self.btn_save)
        
        layout.addLayout(action_layout)

    def create_horizon(self):
        count = len(self.horizons) + 1
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
        color = colors[len(self.horizons) % len(colors)]
        self.horizons.append({'name': f"Horizon_{count}", 'color': color, 'points': [], 'visible': True})
        self.refresh_table(); self.set_active_horizon(len(self.horizons)-1)

    def import_horizon(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv *.txt)")
        if not file_path: return
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1, usecols=[0, 1])
            points = list(zip(df[0], df[1]))
            name = os.path.basename(file_path).split('.')[0]
            colors = ['#FF0000', '#00FF00', '#0000FF']
            color = colors[len(self.horizons) % len(colors)]
            self.horizons.append({'name': name, 'color': color, 'points': points, 'visible': True})
            self.refresh_table(); self.horizon_visibility_changed.emit()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def refresh_table(self):
        self.table.setRowCount(len(self.horizons))
        for btn in self.pick_group.buttons(): self.pick_group.removeButton(btn)
        
        for i, h in enumerate(self.horizons):
            rb = QRadioButton(); rb.setChecked(i == self.active_horizon_index)
            rb.toggled.connect(lambda c, idx=i: self.set_active_horizon(idx) if c else None)
            self.pick_group.addButton(rb)
            widget_rb = QWidget(); l = QHBoxLayout(widget_rb); l.addWidget(rb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 0, widget_rb)
            
            self.table.setItem(i, 1, QTableWidgetItem(h['name']))
            
            btn_col = QPushButton(); btn_col.setStyleSheet(f"background-color: {h['color']}; border: none;")
            btn_col.clicked.connect(lambda _, idx=i: self.change_color(idx))
            self.table.setCellWidget(i, 2, btn_col)
            
            item_pts = QTableWidgetItem(str(len(h['points'])))
            item_pts.setFlags(item_pts.flags() ^ Qt.ItemIsEditable) 
            self.table.setItem(i, 3, item_pts)
            
            btn_del = QPushButton("X"); btn_del.setStyleSheet("color: red; font-weight: bold;")
            btn_del.clicked.connect(lambda _, idx=i: self.delete_horizon(idx))
            self.table.setCellWidget(i, 4, btn_del)
            
        self.btn_pick.setEnabled(len(self.horizons) > 0)

    def set_active_horizon(self, index):
        self.active_horizon_index = index
        if self.is_picking:
            name = self.horizons[index]['name']
            self.lbl_status.setText(f"Status: Picking on {name}"); self.picking_toggled.emit(True, name)

    def toggle_picking(self, checked):
        self.is_picking = checked
        if self.active_horizon_index == -1: self.is_picking = False; self.btn_pick.setChecked(False); return
        name = self.horizons[self.active_horizon_index]['name']
        if self.is_picking:
            self.btn_pick.setText("Stop Picking"); self.btn_pick.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold;")
            self.lbl_status.setText(f"Status: Picking on {name}"); self.lbl_status.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.btn_pick.setText("Start Picking"); self.btn_pick.setStyleSheet("background-color: #e0e0e0;")
            self.lbl_status.setText("Status: Viewing Mode"); self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")
        self.picking_toggled.emit(self.is_picking, name)

    def add_point(self, x, y):
        if self.active_horizon_index == -1: return
        self.horizons[self.active_horizon_index]['points'].append((x, y))
        self.horizons[self.active_horizon_index]['points'].sort(key=lambda p: p[0])
        
        count = len(self.horizons[self.active_horizon_index]['points'])
        item = QTableWidgetItem(str(count))
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(self.active_horizon_index, 3, item)
        
        self.horizon_visibility_changed.emit()

    def delete_closest_point(self, x, y, tolerance_x=10, tolerance_y=50):
        if self.active_horizon_index == -1: return
        
        points = self.horizons[self.active_horizon_index]['points']
        if not points: return
        
        candidates = []
        for i, p in enumerate(points):
            dx = abs(p[0] - x)
            dy = abs(p[1] - y)
            if dx <= tolerance_x and dy <= tolerance_y:
                candidates.append((i, dx + dy))
        
        if candidates:
            candidates.sort(key=lambda k: k[1])
            idx_to_remove = candidates[0][0]
            del points[idx_to_remove]
            
            count = len(points)
            item = QTableWidgetItem(str(count))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(self.active_horizon_index, 3, item)
            
            self.horizon_visibility_changed.emit()

    def change_color(self, index):
        col = QColorDialog.getColor(QColor(self.horizons[index]['color']))
        if col.isValid(): self.horizons[index]['color'] = col.name(); self.refresh_table(); self.horizon_color_changed.emit()

    def delete_horizon(self, index):
        if index == self.active_horizon_index: self.toggle_picking(False); self.active_horizon_index = -1
        del self.horizons[index]; self.refresh_table(); self.horizon_removed.emit()

    def request_export(self):
        if self.active_horizon_index != -1: self.export_requested.emit(self.active_horizon_index)
        else: QMessageBox.warning(self, "Warning", "No horizon selected.")