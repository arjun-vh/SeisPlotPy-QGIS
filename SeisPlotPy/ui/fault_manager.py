import os
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTreeWidget, QTreeWidgetItem, QHeaderView, 
                             QFileDialog, QColorDialog, QCheckBox, QWidget, QMessageBox,
                             QLabel, QRadioButton, QButtonGroup, QComboBox, QInputDialog)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer
import pandas as pd
import os

class FaultManager(QDialog):
    picking_toggled = pyqtSignal(bool, str)
    fault_visibility_changed = pyqtSignal()
    fault_color_changed = pyqtSignal()
    fault_removed = pyqtSignal()
    export_requested = pyqtSignal(int)
    publish_requested = pyqtSignal(int)
    export_all_requested = pyqtSignal()  # Signal for batch export

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fault Interpretation Manager")
        self.resize(750, 450) # Wider for tree view
        
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
        
        # Tree
        self.table = QTreeWidget() # Keep the name self.table to minimize controller disruption
        self.table.setColumnCount(7)
        self.table.setHeaderLabels(["Active", "Vis", "Name", "Group", "Color", "Points", "Actions"])
        
        # Column Resizing
        self.table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Name stretches
        self.table.setColumnWidth(0, 50) # Active radio
        self.table.setColumnWidth(1, 40) # Vis checkbox
        self.table.setColumnWidth(3, 100) # Group
        self.table.setColumnWidth(4, 50) # Color
        self.table.setColumnWidth(5, 50) # Points
        self.table.setColumnWidth(6, 80) # Actions
        
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
        self.faults.append({'name': f"Fault_{count}", 'group': 'Fault', 'color': color, 'points': [], 'visible': True})
        self.refresh_table(); self.set_active_fault(len(self.faults)-1)

    def import_fault(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv *.txt)")
        if not file_path: return
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1, usecols=[0, 1])
            points = list(zip(df[0], df[1]))
            name = os.path.basename(file_path).split('.')[0]
            colors = ['#FF0000', '#00FF00', '#0000FF']
            color = colors[len(self.faults) % len(colors)]
            self.faults.append({'name': name, 'group': 'Fault', 'color': color, 'points': points, 'visible': True})
            self.refresh_table(); self.fault_visibility_changed.emit()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def on_item_changed(self, item, column):
        """Handle edits to tree cells (specifically the Name column)."""
        if column == 2:
            idx = item.data(0, Qt.UserRole)
            if idx is not None and idx < len(self.faults):
                new_name = item.text(2)
                self.faults[idx]['name'] = new_name
                if idx == self.active_fault_index and self.is_picking:
                     self.lbl_status.setText(f"Status: Picking on {new_name}")

    def refresh_table(self):
        self.table.blockSignals(True)
        try:
            self.table.clear()
            self.pick_group = QButtonGroup(self)
            
            groups = set([f.get('group', 'Fault') for f in self.faults])
            
            for group_name in sorted(groups):
                group_item = QTreeWidgetItem(self.table)
                group_item.setText(2, group_name)
                font = group_item.font(2)
                font.setBold(True)
                group_item.setFont(2, font)
                group_item.setExpanded(True)
                
                # Determine group visibility based on children
                children_vis = [f.get('visible', True) for f in self.faults if f.get('group', 'Fault') == group_name]
                group_vis = all(children_vis) if children_vis else True
                
                chk_vis = QCheckBox()
                chk_vis.setChecked(group_vis)
                chk_vis.toggled.connect(lambda c, gn=group_name: self.toggle_group_vis(gn, c))
                w_vis = QWidget(); l2 = QHBoxLayout(w_vis); l2.addWidget(chk_vis); l2.setAlignment(Qt.AlignCenter); l2.setContentsMargins(0,0,0,0)
                self.table.setItemWidget(group_item, 1, w_vis)
                
                btn_col = QPushButton("🎨")
                btn_col.setStyleSheet("background-color: transparent; border: none; font-size: 14px;")
                btn_col.setToolTip("Change color for all faults in group")
                btn_col.clicked.connect(lambda _, gn=group_name: self.change_group_color(gn))
                self.table.setItemWidget(group_item, 4, btn_col)
                
                # Master Map Button
                btn_map_grp = QPushButton("Map")
                btn_map_grp.setStyleSheet("background-color: #e6ffe6; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
                btn_map_grp.setToolTip("Publish all faults in group to QGIS Map Canvas")
                btn_map_grp.clicked.connect(lambda _, gn=group_name: self.publish_group(gn))
                
                grp_action_w = QWidget(); l_ga = QHBoxLayout(grp_action_w); l_ga.setContentsMargins(2,2,2,2); l_ga.addWidget(btn_map_grp)
                self.table.setItemWidget(group_item, 6, grp_action_w)
                
                for i, f in enumerate(self.faults):
                    if f.get('group', 'Fault') != group_name: continue
                    
                    child_item = QTreeWidgetItem(group_item)
                    child_item.setFlags(child_item.flags() | Qt.ItemIsEditable)
                    child_item.setData(0, Qt.UserRole, i) 
                    
                    rb = QRadioButton(); rb.setChecked(i == self.active_fault_index)
                    rb.toggled.connect(lambda c, idx=i: self.set_active_fault(idx) if c else None)
                    self.pick_group.addButton(rb)
                    w_rb = QWidget(); l = QHBoxLayout(w_rb); l.addWidget(rb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
                    self.table.setItemWidget(child_item, 0, w_rb)
                    
                    chk_c = QCheckBox(); chk_c.setChecked(f.get('visible', True))
                    chk_c.toggled.connect(lambda c, idx=i: self.toggle_fault_vis(idx, c))
                    w_c = QWidget(); l3 = QHBoxLayout(w_c); l3.addWidget(chk_c); l3.setAlignment(Qt.AlignCenter); l3.setContentsMargins(0,0,0,0)
                    self.table.setItemWidget(child_item, 1, w_c)
                    
                    child_item.setText(2, f['name'])
                    
                    combo_group = QComboBox()
                    # Do not make it fully editable to prevent 1-letter groups from creating chaos
                    combo_group.addItems(sorted(groups))
                    combo_group.addItem("+ New Group...")
                    combo_group.setCurrentText(group_name)
                    # Use activated instead of currentTextChanged to only fire when user finishes selection
                    combo_group.activated.connect(lambda idx_cb, idx=i, cb=combo_group: self.change_fault_group_combo(idx, cb))
                    self.table.setItemWidget(child_item, 3, combo_group)
                    
                    btn_c = QPushButton(); btn_c.setStyleSheet(f"background-color: {f['color']}; border: none;")
                    btn_c.clicked.connect(lambda _, idx=i: self.change_color(idx))
                    self.table.setItemWidget(child_item, 4, btn_c)
                    
                    child_item.setText(5, str(len(f['points'])))
                    
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(2, 2, 2, 2); action_layout.setSpacing(4)
                    
                    btn_map = QPushButton("Map")
                    btn_map.setStyleSheet("background-color: #e6ffe6; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
                    btn_map.setFixedWidth(40)
                    btn_map.clicked.connect(lambda _, idx=i: self.publish_requested.emit(idx))
                    
                    btn_del = QPushButton("X")
                    btn_del.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
                    btn_del.setFixedWidth(25)
                    btn_del.clicked.connect(lambda _, idx=i: self.delete_fault(idx))
                    
                    action_layout.addWidget(btn_map); action_layout.addWidget(btn_del)
                    self.table.setItemWidget(child_item, 6, action_widget)
            
            self.btn_pick.setEnabled(len(self.faults) > 0)
        finally:
            self.table.blockSignals(False)

    def change_fault_group_combo(self, index, combo_box):
        text = combo_box.currentText()
        if text == "+ New Group...":
            new_group, ok = QInputDialog.getText(self, "New Group", "Enter new group name:")
            if ok and new_group.strip():
                text = new_group.strip()
            else:
                # Revert
                combo_box.setCurrentText(self.faults[index].get('group', 'Fault'))
                return
                
        if not text.strip(): return
        self.faults[index]['group'] = text
        QTimer.singleShot(0, self.refresh_table)
        self.fault_visibility_changed.emit() # Save state

    def publish_group(self, group_name):
        """Emits publish request for all faults in the given group"""
        for i, f in enumerate(self.faults):
            if f.get('group', 'Fault') == group_name:
                self.publish_requested.emit(i)

    def toggle_group_vis(self, group_name, state):
        changed = False
        for f in self.faults:
            if f.get('group', 'Fault') == group_name:
                f['visible'] = state
                changed = True
        if changed:
            self.refresh_table()
            self.fault_visibility_changed.emit()

    def change_group_color(self, group_name):
        col = QColorDialog.getColor(QColor('#FF0000'), self)
        if col.isValid():
            c_name = col.name()
            changed = False
            for f in self.faults:
                if f.get('group', 'Fault') == group_name:
                    f['color'] = c_name
                    changed = True
            if changed:
                self.refresh_table()
                self.fault_color_changed.emit()

    def toggle_fault_vis(self, index, state):
        self.faults[index]['visible'] = state
        self.refresh_table()
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

    def _find_tree_item(self, target_idx):
        for i in range(self.table.topLevelItemCount()):
            group_item = self.table.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                if child.data(0, Qt.UserRole) == target_idx:
                    return child
        return None

    def add_point(self, x, y):
        if self.active_fault_index == -1: return
        # CRITICAL DIFFERENCE: APPEND ONLY, NO SORTING for Faults
        self.faults[self.active_fault_index]['points'].append((x, y))
        
        count = len(self.faults[self.active_fault_index]['points'])
        item = self._find_tree_item(self.active_fault_index)
        if item:
            item.setText(5, str(count))
        
        self.fault_visibility_changed.emit()

    def delete_closest_point(self, x, y, tolerance_x=10, tolerance_y=50):
        if self.active_fault_index == -1: return
        
        points = self.faults[self.active_fault_index]['points']
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
            item = self._find_tree_item(self.active_fault_index)
            if item:
                item.setText(5, str(count))
            
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
        for f in self.faults:
            f.setdefault('group', 'Fault')
        self.refresh_table()
