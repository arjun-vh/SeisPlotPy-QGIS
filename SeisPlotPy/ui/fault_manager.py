import os
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QFileDialog, QColorDialog, QCheckBox, QWidget, QMessageBox,
                             QLabel, QRadioButton, QButtonGroup)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, pyqtSignal
import pandas as pd

class FaultManager(QDialog):
    picking_toggled = pyqtSignal(bool, str)
    fault_visibility_changed = pyqtSignal()
    fault_color_changed = pyqtSignal()
    fault_removed = pyqtSignal()
    export_requested = pyqtSignal(int)
    publish_requested = pyqtSignal(int)
    export_all_requested = pyqtSignal()  # Signal for batch export # Signal for "Map" button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fault Interpretation Manager")
        self.resize(650, 400)
        
        self.setWindowFlags(Qt.Window)
        
        self.faults = []
        self.active_fault_index = -1
        self.is_picking = False
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton("+ New Fault"); self.btn_new.clicked.connect(self.create_fault)
        self.btn_import = QPushButton("Import CSV"); self.btn_import.clicked.connect(self.import_fault)
        btn_layout.addWidget(self.btn_new); btn_layout.addWidget(self.btn_import)
        layout.addLayout(btn_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Active", "Vis", "Name", "Color", "Points", "Actions"])
        
        # Column Resizing
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 40)
        self.table.setColumnWidth(3, 50)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(5, 80)
        
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
        
        self.btn_save_all = QPushButton("Export All Visible")
        self.btn_save_all.clicked.connect(lambda: self.export_all_requested.emit())
        action_layout.addWidget(self.btn_save_all)
        
        layout.addLayout(action_layout)

    def create_fault(self):
        count = len(self.faults) + 1
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
        color = colors[len(self.faults) % len(colors)]
        self.faults.append({'name': f"Fault_{count}", 'color': color, 'points': [], 'visible': True})
        self.refresh_table(); self.set_active_fault(len(self.faults)-1)

    def import_fault(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv *.txt)")
        if not file_path: return
        try:
            # Assumes Column 0 is X, Column 1 is Y/Time. Presereves order!
            df = pd.read_csv(file_path, header=None, skiprows=1, usecols=[0, 1])
            points = list(zip(df[0], df[1]))
            name = os.path.basename(file_path).split('.')[0]
            colors = ['#FF0000', '#00FF00', '#0000FF']
            color = colors[len(self.faults) % len(colors)]
            self.faults.append({'name': name, 'color': color, 'points': points, 'visible': True})
            self.refresh_table(); self.fault_visibility_changed.emit()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def on_item_changed(self, item):
        if item.column() == 2:
            row = item.row()
            if row < len(self.faults):
                new_name = item.text()
                self.faults[row]['name'] = new_name
                if row == self.active_fault_index and self.is_picking:
                     self.lbl_status.setText(f"Status: Picking on {new_name}")

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.faults))
        
        for btn in self.pick_group.buttons(): self.pick_group.removeButton(btn)
        
        for i, f in enumerate(self.faults):
            # 0. Active
            rb = QRadioButton(); rb.setChecked(i == self.active_fault_index)
            rb.toggled.connect(lambda c, idx=i: self.set_active_fault(idx) if c else None)
            self.pick_group.addButton(rb)
            w_rb = QWidget(); l = QHBoxLayout(w_rb); l.addWidget(rb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 0, w_rb)
            
            # 1. Vis
            chk_vis = QCheckBox()
            chk_vis.setChecked(f.get('visible', True))
            chk_vis.toggled.connect(lambda c, idx=i: self.toggle_fault_vis(idx, c))
            w_vis = QWidget(); l2 = QHBoxLayout(w_vis); l2.addWidget(chk_vis); l2.setAlignment(Qt.AlignCenter); l2.setContentsMargins(0,0,0,0)
            self.table.setCellWidget(i, 1, w_vis)

            # 2. Name
            self.table.setItem(i, 2, QTableWidgetItem(f['name']))
            
            # 3. Color
            btn_col = QPushButton(); btn_col.setStyleSheet(f"background-color: {f['color']}; border: none;")
            btn_col.clicked.connect(lambda _, idx=i: self.change_color(idx))
            self.table.setCellWidget(i, 3, btn_col)
            
            # 4. Points
            item_pts = QTableWidgetItem(str(len(f['points'])))
            item_pts.setFlags(item_pts.flags() ^ Qt.ItemIsEditable) 
            self.table.setItem(i, 4, item_pts)
            
            # 5. Actions
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_layout.setSpacing(4)
            
            btn_map = QPushButton("Map")
            btn_map.setToolTip("Publish fault to QGIS")
            btn_map.setStyleSheet("background-color: #e6ffe6; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
            btn_map.setFixedWidth(40)
            btn_map.clicked.connect(lambda _, idx=i: self.publish_requested.emit(idx))
            
            btn_del = QPushButton("X")
            btn_del.setToolTip("Delete Fault")
            btn_del.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
            btn_del.setFixedWidth(25)
            btn_del.clicked.connect(lambda _, idx=i: self.delete_fault(idx))
            
            action_layout.addWidget(btn_map)
            action_layout.addWidget(btn_del)
            self.table.setCellWidget(i, 5, action_widget)
            
        self.btn_pick.setEnabled(len(self.faults) > 0)
        self.table.blockSignals(False)

    def toggle_fault_vis(self, index, state):
        self.faults[index]['visible'] = state
        self.fault_visibility_changed.emit()

    def set_active_fault(self, index):
        self.active_fault_index = index
        if self.is_picking:
            name = self.faults[index]['name']
            self.lbl_status.setText(f"Status: Picking on {name}"); self.picking_toggled.emit(True, name)

    def toggle_picking(self, checked):
        self.is_picking = checked
        if self.active_fault_index == -1: self.is_picking = False; self.btn_pick.setChecked(False); return
        name = self.faults[self.active_fault_index]['name']
        if self.is_picking:
            self.btn_pick.setText("Stop Picking"); self.btn_pick.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold;")
            self.lbl_status.setText(f"Status: Picking on {name}"); self.lbl_status.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.btn_pick.setText("Start Picking"); self.btn_pick.setStyleSheet("background-color: #e0e0e0;")
            self.lbl_status.setText("Status: Viewing Mode"); self.lbl_status.setStyleSheet("font-weight: bold; color: gray;")
        self.picking_toggled.emit(self.is_picking, name)

    def add_point(self, x, y):
        if self.active_fault_index == -1: return
        # CRITICAL DIFFERENCE: APPEND ONLY, NO SORTING
        self.faults[self.active_fault_index]['points'].append((x, y))
        
        count = len(self.faults[self.active_fault_index]['points'])
        item = QTableWidgetItem(str(count))
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(self.active_fault_index, 4, item)
        
        self.fault_visibility_changed.emit()



    def change_color(self, index):
        col = QColorDialog.getColor(QColor(self.faults[index]['color']), self)
        if col.isValid(): self.faults[index]['color'] = col.name(); self.refresh_table(); self.fault_color_changed.emit()

    def delete_fault(self, index):
        if index == self.active_fault_index: self.toggle_picking(False); self.active_fault_index = -1
        del self.faults[index]; self.refresh_table(); self.fault_removed.emit()

    def request_export(self):
        if self.active_fault_index != -1: self.export_requested.emit(self.active_fault_index)
        else: QMessageBox.warning(self, "Warning", "No fault selected.")

    def get_state(self):
        return self.faults

    def restore_state(self, faults_data):
        if not faults_data: return
        self.faults = faults_data
        self.refresh_table()
