import os
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTreeWidget, QTreeWidgetItem, QHeaderView, 
                             QFileDialog, QColorDialog, QCheckBox, QWidget, QMessageBox,
                             QLabel, QRadioButton, QButtonGroup, QComboBox)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt, pyqtSignal
import pandas as pd
import os

class HorizonManager(QDialog):
    picking_toggled = pyqtSignal(bool, str)
    horizon_visibility_changed = pyqtSignal()
    horizon_color_changed = pyqtSignal()
    horizon_removed = pyqtSignal()
    export_requested = pyqtSignal(int)
    publish_requested = pyqtSignal(int) # Signal for "Map" button
    export_all_requested = pyqtSignal()  # Signal for batch export
    flatten_toggled = pyqtSignal(int, bool)  # Signal for flatten: (horizon_index, is_active)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Horizon Interpretation Manager")
        self.resize(750, 450) # Wider for tree view
        
        self.setWindowFlags(Qt.Window)
        
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
        
        # Tree
        self.table = QTreeWidget() # Keep the name self.table to minimize controller disruption
        self.table.setColumnCount(8)
        self.table.setHeaderLabels(["Active", "Vis", "Name", "Group", "Color", "Points", "Flat", "Actions"])
        
        # Column Resizing
        self.table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Name stretches
        self.table.setColumnWidth(0, 50) # Active radio
        self.table.setColumnWidth(1, 40) # Vis checkbox
        self.table.setColumnWidth(3, 100) # Group
        self.table.setColumnWidth(4, 50) # Color
        self.table.setColumnWidth(5, 50) # Points
        self.table.setColumnWidth(6, 45) # Flat
        self.table.setColumnWidth(7, 80) # Actions
        
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

    def create_horizon(self):
        count = len(self.horizons) + 1
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
        color = colors[len(self.horizons) % len(colors)]
        self.horizons.append({'name': f"Horizon_{count}", 'group': 'Horizon', 'color': color, 'points': [], 'visible': True, 'flattened': False})
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
            self.horizons.append({'name': name, 'group': 'Horizon', 'color': color, 'points': points, 'visible': True, 'flattened': False})
            self.refresh_table(); self.horizon_visibility_changed.emit()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def on_item_changed(self, item, column):
        """Handle edits to tree cells (specifically the Name column)."""
        # Column 2 is Name
        if column == 2:
            idx = item.data(0, Qt.UserRole)
            if idx is not None and idx < len(self.horizons):
                new_name = item.text(2)
                self.horizons[idx]['name'] = new_name
                if idx == self.active_horizon_index and self.is_picking:
                     self.lbl_status.setText(f"Status: Picking on {new_name}")

    def refresh_table(self):
        self.table.blockSignals(True)
        try:
            self.table.clear()
            self.pick_group = QButtonGroup(self)
            
            groups = set([h.get('group', 'Horizon') for h in self.horizons])
            
            for group_name in sorted(groups):
                group_item = QTreeWidgetItem(self.table)
                group_item.setText(2, group_name)
                font = group_item.font(2)
                font.setBold(True)
                group_item.setFont(2, font)
                group_item.setExpanded(True)
                
                # Determine group visibility based on children
                children_vis = [h.get('visible', True) for h in self.horizons if h.get('group', 'Horizon') == group_name]
                group_vis = all(children_vis) if children_vis else True
                
                chk_vis = QCheckBox()
                chk_vis.setChecked(group_vis)
                chk_vis.toggled.connect(lambda c, gn=group_name: self.toggle_group_vis(gn, c))
                w_vis = QWidget(); l2 = QHBoxLayout(w_vis); l2.addWidget(chk_vis); l2.setAlignment(Qt.AlignCenter); l2.setContentsMargins(0,0,0,0)
                self.table.setItemWidget(group_item, 1, w_vis)
                
                btn_col = QPushButton("🎨")
                btn_col.setStyleSheet("background-color: transparent; border: none; font-size: 14px;")
                btn_col.setToolTip("Change color for all horizons in group")
                btn_col.clicked.connect(lambda _, gn=group_name: self.change_group_color(gn))
                self.table.setItemWidget(group_item, 4, btn_col)
                
                # Master Map Button
                btn_map_grp = QPushButton("Map")
                btn_map_grp.setStyleSheet("background-color: #e6ffe6; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
                btn_map_grp.setToolTip("Publish all horizons in group to QGIS Map Canvas")
                btn_map_grp.clicked.connect(lambda _, gn=group_name: self.publish_group(gn))
                
                grp_action_w = QWidget(); l_ga = QHBoxLayout(grp_action_w); l_ga.setContentsMargins(2,2,2,2); l_ga.addWidget(btn_map_grp)
                self.table.setItemWidget(group_item, 7, grp_action_w)
                
                for i, h in enumerate(self.horizons):
                    if h.get('group', 'Horizon') != group_name: continue
                    
                    child_item = QTreeWidgetItem(group_item)
                    child_item.setFlags(child_item.flags() | Qt.ItemIsEditable)
                    child_item.setData(0, Qt.UserRole, i) 
                    
                    rb = QRadioButton(); rb.setChecked(i == self.active_horizon_index)
                    rb.toggled.connect(lambda c, idx=i: self.set_active_horizon(idx) if c else None)
                    self.pick_group.addButton(rb)
                    w_rb = QWidget(); l = QHBoxLayout(w_rb); l.addWidget(rb); l.setAlignment(Qt.AlignCenter); l.setContentsMargins(0,0,0,0)
                    self.table.setItemWidget(child_item, 0, w_rb)
                    
                    chk_c = QCheckBox(); chk_c.setChecked(h.get('visible', True))
                    chk_c.toggled.connect(lambda c, idx=i: self.toggle_horizon_vis(idx, c))
                    w_c = QWidget(); l3 = QHBoxLayout(w_c); l3.addWidget(chk_c); l3.setAlignment(Qt.AlignCenter); l3.setContentsMargins(0,0,0,0)
                    self.table.setItemWidget(child_item, 1, w_c)
                    
                    child_item.setText(2, h['name'])
                    
                    combo_group = QComboBox()
                    # Do not make it fully editable to prevent 1-letter groups from creating chaos
                    combo_group.addItems(sorted(groups))
                    combo_group.addItem("+ New Group...")
                    combo_group.setCurrentText(group_name)
                    # Use activated instead of currentTextChanged to only fire when user finishes selection
                    combo_group.activated.connect(lambda idx_cb, idx=i, cb=combo_group: self.change_horizon_group_combo(idx, cb))
                    self.table.setItemWidget(child_item, 3, combo_group)
                    
                    btn_c = QPushButton(); btn_c.setStyleSheet(f"background-color: {h['color']}; border: none;")
                    btn_c.clicked.connect(lambda _, idx=i: self.change_color(idx))
                    self.table.setItemWidget(child_item, 4, btn_c)
                    
                    child_item.setText(5, str(len(h['points'])))
                    
                    btn_flat = QPushButton("Flat")
                    btn_flat.setCheckable(True)
                    is_flat = h.get('flattened', False)
                    btn_flat.setChecked(is_flat)
                    if is_flat: btn_flat.setStyleSheet("background-color: #ffd700; border: 1px solid #aaa; border-radius: 3px; font-size: 10px; font-weight: bold;")
                    else: btn_flat.setStyleSheet("background-color: #f0f0f0; border: 1px solid #aaa; border-radius: 3px; font-size: 10px;")
                    btn_flat.setFixedWidth(35)
                    btn_flat.toggled.connect(lambda c, idx=i: self.toggle_flatten(idx, c))
                    self.table.setItemWidget(child_item, 6, btn_flat)
                    
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
                    btn_del.clicked.connect(lambda _, idx=i: self.delete_horizon(idx))
                    
                    action_layout.addWidget(btn_map); action_layout.addWidget(btn_del)
                    self.table.setItemWidget(child_item, 7, action_widget)
            
            self.btn_pick.setEnabled(len(self.horizons) > 0)
        finally:
            self.table.blockSignals(False)

    def change_horizon_group_combo(self, index, combo_box):
        text = combo_box.currentText()
        if text == "+ New Group...":
            from qgis.PyQt.QtWidgets import QInputDialog
            new_group, ok = QInputDialog.getText(self, "New Group", "Enter new group name:")
            if ok and new_group.strip():
                text = new_group.strip()
            else:
                # Revert
                combo_box.setCurrentText(self.horizons[index].get('group', 'Horizon'))
                return
                
        if not text.strip(): return
        self.horizons[index]['group'] = text
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self.refresh_table)
        self.horizon_visibility_changed.emit() # Save state

    def publish_group(self, group_name):
        """Emits publish request for all horizons in the given group"""
        for i, h in enumerate(self.horizons):
            if h.get('group', 'Horizon') == group_name:
                self.publish_requested.emit(i)



    def toggle_group_vis(self, group_name, state):
        changed = False
        for h in self.horizons:
            if h.get('group', 'Horizon') == group_name:
                h['visible'] = state
                changed = True
        if changed:
            self.refresh_table()
            self.horizon_visibility_changed.emit()

    def change_group_color(self, group_name):
        col = QColorDialog.getColor(QColor('#FF0000'), self)
        if col.isValid():
            c_name = col.name()
            changed = False
            for h in self.horizons:
                if h.get('group', 'Horizon') == group_name:
                    h['color'] = c_name
                    changed = True
            if changed:
                self.refresh_table()
                self.horizon_color_changed.emit()

    def toggle_horizon_vis(self, index, state):
        self.horizons[index]['visible'] = state
        self.refresh_table()
        self.horizon_visibility_changed.emit()

    def toggle_flatten(self, index, state):
        # Enforce radio-behavior (only one flat at a time)
        for i, h in enumerate(self.horizons):
            if i == index:
                h['flattened'] = state
            else:
                h['flattened'] = False
        self.refresh_table()
        self.flatten_toggled.emit(index, state)

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

    def _find_tree_item(self, target_idx):
        for i in range(self.table.topLevelItemCount()):
            group_item = self.table.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                if child.data(0, Qt.UserRole) == target_idx:
                    return child
        return None

    def add_point(self, x, y):
        if self.active_horizon_index == -1: return
        self.horizons[self.active_horizon_index]['points'].append((x, y))
        self.horizons[self.active_horizon_index]['points'].sort(key=lambda p: p[0])
        
        count = len(self.horizons[self.active_horizon_index]['points'])
        item = self._find_tree_item(self.active_horizon_index)
        if item:
            item.setText(5, str(count))
        
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
            item = self._find_tree_item(self.active_horizon_index)
            if item:
                item.setText(5, str(count))
            
            self.horizon_visibility_changed.emit()



    def change_color(self, index):
        col = QColorDialog.getColor(QColor(self.horizons[index]['color']), self)
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
        for h in self.horizons:
            h.setdefault('group', 'Horizon')
        self.refresh_table()