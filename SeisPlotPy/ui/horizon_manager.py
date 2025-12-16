import os
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QFileDialog, QColorDialog, QCheckBox, QWidget, QMessageBox,
                             QLabel, QRadioButton, QButtonGroup)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, pyqtSignal
import pandas as pd

class HorizonManager(QDialog):
    picking_toggled = pyqtSignal(bool, str)
    horizon_visibility_changed = pyqtSignal()
    horizon_color_changed = pyqtSignal()
    horizon_removed = pyqtSignal()
    export_requested = pyqtSignal(int)
    publish_requested = pyqtSignal(int) # Signal for "Map" button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Horizon Interpretation Manager")
        self.resize(650, 400) # Slightly wider for new column
        
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
        # Increased columns to 6 to add "Vis" (Visibility)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Active", "Vis", "Name", "Color", "Points", "Actions"])
        
        # Column Resizing
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Name stretches
        self.table.setColumnWidth(0, 50) # Active radio
        self.table.setColumnWidth(1, 40) # Vis checkbox
        self.table.setColumnWidth(3, 50) # Color
        self.table.setColumnWidth(4, 60) # Points
        self.table.setColumnWidth(5, 80) # Actions
        
        # --- CRITICAL FIX: Connect item changed signal to save Name edits ---
        self.table.itemChanged.connect(self.on_item_changed)
        
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

    def on_item_changed(self, item):
        """Handle edits to table cells (specifically the Name column)."""
        # Column 2 is Name
        if item.column() == 2:
            row = item.row()
            if row < len(self.horizons):
                new_name = item.text()
                self.horizons[row]['name'] = new_name
                # If currently picking this horizon, update the status label
                if row == self.active_horizon_index and self.is_picking:
                     self.lbl_status.setText(f"Status: Picking on {new_name}")

    def refresh_table(self):
        # Block signals to prevent 'on_item_changed' from firing while we build the table
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.horizons))
        
        for btn in self.pick_group.buttons(): self.pick_group.removeButton(btn)
        
        for i, h in enumerate(self.horizons):
            # 0. Active Radio Button
            rb = QRadioButton(); rb.setChecked(i == self.active_horizon_index)
            rb.toggled.connect(lambda c, idx=i: self.set_active_horizon(idx) if c else None)
            self.pick_group.addButton(rb)
            w_rb = QWidget(); l = QHBoxLayout(w_rb); l.addWidget(rb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 0, w_rb)
            
            # 1. NEW: Visibility Checkbox
            chk_vis = QCheckBox()
            chk_vis.setChecked(h.get('visible', True))
            # Connect toggle to data update
            chk_vis.toggled.connect(lambda c, idx=i: self.toggle_horizon_vis(idx, c))
            w_vis = QWidget(); l2 = QHBoxLayout(w_vis); l2.addWidget(chk_vis); l2.setAlignment(Qt.AlignCenter); l2.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 1, w_vis)

            # 2. Name
            self.table.setItem(i, 2, QTableWidgetItem(h['name']))
            
            # 3. Color
            btn_col = QPushButton(); btn_col.setStyleSheet(f"background-color: {h['color']}; border: none;")
            btn_col.clicked.connect(lambda _, idx=i: self.change_color(idx))
            self.table.setCellWidget(i, 3, btn_col)
            
            # 4. Point Count
            item_pts = QTableWidgetItem(str(len(h['points'])))
            item_pts.setFlags(item_pts.flags() ^ Qt.ItemIsEditable) 
            self.table.setItem(i, 4, item_pts)
            
            # 5. Actions (Map + Delete)
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_layout.setSpacing(4)
            
            btn_map = QPushButton("Map")
            btn_map.setToolTip("Publish horizon as a layer to QGIS Map Canvas")
            btn_map.setStyleSheet("background-color: #e6ffe6; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
            btn_map.setFixedWidth(40)
            btn_map.clicked.connect(lambda _, idx=i: self.publish_requested.emit(idx))
            
            btn_del = QPushButton("X")
            btn_del.setToolTip("Delete Horizon")
            btn_del.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
            btn_del.setFixedWidth(25)
            btn_del.clicked.connect(lambda _, idx=i: self.delete_horizon(idx))
            
            action_layout.addWidget(btn_map)
            action_layout.addWidget(btn_del)
            self.table.setCellWidget(i, 5, action_widget)
            
        self.btn_pick.setEnabled(len(self.horizons) > 0)
        self.table.blockSignals(False)

    def toggle_horizon_vis(self, index, state):
        self.horizons[index]['visible'] = state
        self.horizon_visibility_changed.emit()

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
        self.table.setItem(self.active_horizon_index, 4, item) # Updated col index
        
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
            self.table.setItem(self.active_horizon_index, 4, item) # Updated col index
            
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

    def get_state(self):
        return self.horizons

    def restore_state(self, horizons_data):
        if not horizons_data: return
        self.horizons = horizons_data
        self.refresh_table()