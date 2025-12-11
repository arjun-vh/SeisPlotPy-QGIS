import os
from qgis.PyQt.QtWidgets import (QApplication, QFileDialog, QMessageBox, QInputDialog, 
                             QMenuBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QDoubleSpinBox, QDialog, QDialogButtonBox, 
                             QComboBox, QRadioButton, QButtonGroup, QAction)
from qgis.PyQt.QtWidgets import (QApplication, QFileDialog, QMessageBox, QInputDialog, 
                             QMenuBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QDoubleSpinBox, QDialog, QDialogButtonBox, 
                             QComboBox, QRadioButton, QButtonGroup, QAction)
from qgis.PyQt.QtGui import QCursor, QIcon, QColor, QPainter
from qgis.PyQt.QtCore import Qt, QVariant
import json

# --- QGIS IMPORTS ---
from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, 
                       QgsPointXY, QgsProject, QgsField, 
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsWkbTypes, QgsSingleSymbolRenderer, QgsSymbol, QgsSimpleLineSymbolLayer)
from qgis.gui import QgsProjectionSelectionDialog, QgsRubberBand


# --- SPATIAL SEARCH IMPORT ---
from scipy.spatial import cKDTree


import pyqtgraph as pg

# Internal Imports
from .ui.seismic_view import SeismicView
from .ui.header_tools import TextHeaderDialog, HeaderQCPlot, SpectrumPlot, HeaderExportDialog
from .ui.horizon_manager import HorizonManager
from .ui.dialogs import GeometryDialog, BandpassDialog
from .core.data_handler import SeismicDataManager
from .core.processing import SeismicProcessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MainController:
    def __init__(self, iface):
        self.iface = iface 
        self.view = SeismicView()
        self.data_manager = None
        self.current_data = None 
        
        self.active_header_map = None
        self.full_cum_dist = None 
        self.dist_unit = "m" 
        
        self.is_programmatic_update = False
        self.is_picking_mode = False 
        
        self.horizon_manager = HorizonManager(None)
        self.horizon_items = [] 
        
        # --- Navigation Attributes ---
        self.coord_tree = None      
        self.map_marker = None
        self.qgis_layer = None 
        self.view_highlight = None 
        self.world_coords = None
        self.coord_step = 1  # Track decimation step for coordinate array   
        

        # --- Hook into the Window Close Event ---
        self.view.closeEvent = self.cleanup_on_close
        
        # Auto-save horizons on change
        self.horizon_manager.horizon_visibility_changed.connect(self.save_horizons)
        self.horizon_manager.horizon_removed.connect(self.save_horizons)
        self.horizon_manager.horizon_color_changed.connect(self.save_horizons)
        # We need to capture when a point is added too, which emits visibility changed? Yes, checked source.

        

        self.setup_menu()

        # Connections
        self.view.btn_load.clicked.connect(self.load_file)
        self.view.btn_apply.clicked.connect(self.apply_changes)
        self.view.btn_reset.clicked.connect(self.reset_view)
        
        self.view.chk_manual_step.stateChanged.connect(self.toggle_manual_step)
        self.view.combo_cmap.currentTextChanged.connect(self.change_colormap)
        self.view.spin_contrast.valueChanged.connect(self.update_contrast)
        self.view.combo_domain.currentTextChanged.connect(self.update_labels)
        self.view.btn_export.clicked.connect(self.export_figure)
        self.view.chk_flip_x.stateChanged.connect(self.toggle_flip_x)
        self.view.chk_grid.stateChanged.connect(self.toggle_grid)
        self.view.chk_smooth.stateChanged.connect(self.toggle_smooth)
        self.view.btn_preview_ratio.clicked.connect(self.match_aspect_ratio)
        self.view.plot_widget.sigRangeChanged.connect(self.sync_view_to_controls)
        self.view.combo_header.activated.connect(self.on_header_changed)
        
        self.horizon_manager.picking_toggled.connect(self.set_picking_mode)
        self.horizon_manager.horizon_visibility_changed.connect(self.draw_horizons)
        self.horizon_manager.horizon_color_changed.connect(self.draw_horizons)
        self.horizon_manager.horizon_removed.connect(self.draw_horizons)
        
        # This line was causing the error because the method was missing
        self.horizon_manager.export_requested.connect(self.handle_horizon_export)
        self.horizon_manager.publish_requested.connect(self.publish_horizon_to_map)
        
        self.view.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        
        self.view.show()

    # =========================================================================
    # --- CLEANUP & MAP INTERACTION ---
    # =========================================================================
    def cleanup_on_close(self, event):
        """Called when the user closes the Seismic Window."""
        self.save_horizons() # Ensure saved
        
        # Ensure latest state is on the layer
        self.update_layer_state()

        if self.view_highlight:
            self.view_highlight.reset(QgsWkbTypes.LineGeometry)
            self.view_highlight = None
        
        # CRITICAL CHANGE: Do NOT remove the layer from QGIS Project.
        # The layer persists so it can be double-clicked later to re-open this window.
        
        event.accept()

    
    def on_layer_deleted(self):
        """Called automatically if the user deletes the layer from QGIS Layer Panel."""
        self.qgis_layer = None
        # Also clear the red highlight line if it exists
        if self.view_highlight:
            self.view_highlight.reset(QgsWkbTypes.LineGeometry)
            self.view_highlight = None
    # 

    def create_qgis_layer(self, x_coords, y_coords, crs=None, geom_params=None):
        """Creates QGIS layer AND builds spatial index for navigation."""
        layer_name = os.path.basename(self.data_manager.file_path)
        
        self.world_coords = np.column_stack((x_coords, y_coords))
        self.coord_tree = cKDTree(self.world_coords)
        # For fresh loads, assume full resolution (step=1) since calculate_distance uses step=1
        self.coord_step = 1
        self.view.update_status(f"Navigation Index Built: {len(x_coords)} points")

        crs_def = ""
        if crs is not None and crs.isValid():
            crs_def = f"?crs={crs.authid()}"

        layer = QgsVectorLayer(f"LineString{crs_def}", layer_name, "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("filename", QVariant.String), QgsField("traces", QVariant.Int)])
        layer.updateFields()

        points = [QgsPointXY(float(x), float(y)) for x, y in zip(x_coords, y_coords)]
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolylineXY(points))
        feat.setAttributes([layer_name, len(x_coords)])
        
        pr.addFeatures([feat])
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)
        
        self.qgis_layer = layer
        
        # Connect signal to handle deletion safely ---
        self.qgis_layer.willBeDeleted.connect(self.on_layer_deleted)
        # ---------------------------------------------------
        
        if self.iface.mapCanvas():
            # Transform extent to Map CRS if needed
            map_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_extent = layer.extent()
            
            if crs is not None and crs.isValid() and map_crs != crs:
                try:
                    xform = QgsCoordinateTransform(crs, map_crs, QgsProject.instance())
                    map_extent = xform.transformBoundingBox(layer_extent)
                    self.iface.mapCanvas().setExtent(map_extent)
                except:
                    self.iface.mapCanvas().setExtent(layer_extent)
            else:
                self.iface.mapCanvas().setExtent(layer_extent)
                
            self.iface.mapCanvas().refresh()
            
        # Metadata for Persistence
        layer.setCustomProperty("seisplotpy_path", self.data_manager.file_path)
        
        # SAVE GEOMETRY PARAMS (Headers/Scalar used)
        # Consolidate on GeometryDialog format
        import json
        if geom_params:
            layer.setCustomProperty("seisplotpy_geometry_params", json.dumps(geom_params))
        else:
            # Fallback default
            params = {"x_key": "CDP_X", "y_key": "CDP_Y", "use_header": True, "scalar_key": "Source_Group_Scalar"}
            layer.setCustomProperty("seisplotpy_geometry_params", json.dumps(params))
        
        self.update_layer_state()

    def update_layer_state(self):
        """Updates the JSON state stored on the layer custom properties."""
        if self.qgis_layer and self.qgis_layer.isValid():
            try:
                state = self.get_state()
                if state:
                    self.qgis_layer.setCustomProperty("seisplotpy_state", json.dumps(state))
            except: pass

    def _transform_mouse_point(self, point):
        """Helper to transform Map Canvas Point -> Layer CRS Point."""
        if self.qgis_layer is None: return point
        
        # --- FIX: Added Try-Except block for RuntimeErrors ---
        try:
            # Check if the C++ object is still valid
            if not self.qgis_layer.isValid():
                self.qgis_layer = None
                return point

            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = self.qgis_layer.crs() 

            if canvas_crs != layer_crs and layer_crs.isValid():
                xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
                return xform.transform(point)
        except (RuntimeError, Exception):
            # If the layer was deleted, this catches the error and prevents the crash
            self.qgis_layer = None
            return point
        # ---------------------------------------------------
            
        return point

    def handle_map_hover(self, point):
        """Received from SeisPlotPy when mouse moves on canvas."""
        if self.coord_tree is None: return
        
        search_point = self._transform_mouse_point(point)
        dist, idx = self.coord_tree.query([search_point.x(), search_point.y()])

        # Calculate tolerance in LAYER UNITS
        # MapUnitsPerPixel is in Map CRS (e.g. Degrees)
        # We need to know how big a pixel is in Layer CRS (e.g. Meters)
        
        map_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        layer_crs = self.qgis_layer.crs() if self.qgis_layer else map_crs
        
        pixel_size_map = self.iface.mapCanvas().mapUnitsPerPixel()
        tolerance = pixel_size_map * 20 # Default fallback
        
        if map_crs != layer_crs and layer_crs.isValid():
            try:
                # Approximate 1 pixel vector transformation
                # We interpret map_units_per_pixel as a generic scale factor
                # Better: Measure 1 pixel distance at the point location
                pt_map = point
                pt_map_plus_10px = QgsPointXY(point.x() + pixel_size_map * 10, point.y())
                
                xform = QgsCoordinateTransform(map_crs, layer_crs, QgsProject.instance())
                pt_layer = xform.transform(pt_map)
                pt_layer_plus = xform.transform(pt_map_plus_10px)
                
                # Distance in layer units for 10 pixels
                dist_layer = np.sqrt(pt_layer.sqrDist(pt_layer_plus))
                tolerance = dist_layer * 2 # 20 pixels total tolerance (10px * 2)
            except: pass

        if dist > tolerance: 
            if self.map_marker: self.map_marker.hide()
            return
            
        real_idx = idx * self.coord_step
        plot_x_value = 0
        current_header = self.view.combo_header.currentText()
        
        if current_header == "Trace Index":
            plot_x_value = real_idx
        elif self.active_header_map is not None and real_idx < len(self.active_header_map):
            plot_x_value = self.active_header_map[real_idx]
        else:
            plot_x_value = real_idx

        self.update_map_marker(plot_x_value)

    def handle_map_click(self, point):
        """Received from SeisPlotPy on double-click."""
        if self.coord_tree is None: return
        
        search_point = self._transform_mouse_point(point)
        dist, _ = self.coord_tree.query([search_point.x(), search_point.y()])
        
        # Tolerance logic (Duplicated from hover - should refactor but this is safe)
        map_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        layer_crs = self.qgis_layer.crs() if self.qgis_layer else map_crs
        pixel_size_map = self.iface.mapCanvas().mapUnitsPerPixel()
        tolerance = pixel_size_map * 10 
        
        if map_crs != layer_crs and layer_crs.isValid():
            try:
                pt_map = point
                pt_map_plus = QgsPointXY(point.x() + pixel_size_map * 10, point.y())
                xform = QgsCoordinateTransform(map_crs, layer_crs, QgsProject.instance())
                dist_layer = np.sqrt(xform.transform(pt_map).sqrDist(xform.transform(pt_map_plus)))
                tolerance = dist_layer
            except: pass
        
        if dist < tolerance:
            self.view.showNormal() 
            self.view.show()       
            self.view.raise_()     
            self.view.activateWindow() 

    def update_map_marker(self, x_pos):
        if self.map_marker is None:
            self.map_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2, style=Qt.DashLine))
            self.view.plot_widget.addItem(self.map_marker)
        
        self.map_marker.setValue(x_pos)
        self.map_marker.show()

    def highlight_map_extent(self, x_range):
        if self.coord_tree is None or self.world_coords is None: return
        
        # Guard against deleted layer
        if self.qgis_layer is None: 
            if self.view_highlight: self.view_highlight.hide()
            return
            
        # --- FIX: Additional check for validity ---
        try:
            if not self.qgis_layer.isValid():
                self.qgis_layer = None
                if self.view_highlight: self.view_highlight.hide()
                return
        except RuntimeError:
            self.qgis_layer = None
            return
        # ------------------------------------------

        min_val, max_val = x_range
        start_idx, end_idx = 0, 0
        
        header = self.view.combo_header.currentText()
        n_points = self.world_coords.shape[0]

        if header == "Trace Index":
            # Account for coordinate decimation
            # If coord_step=10, then trace 100 maps to world_coords[10]
            start_idx = int(min_val / self.coord_step)
            end_idx = int(max_val / self.coord_step)
        elif self.active_header_map is not None:
            try:
                if self.active_header_map[0] < self.active_header_map[-1]:
                    start_idx = np.searchsorted(self.active_header_map, min_val)
                    end_idx = np.searchsorted(self.active_header_map, max_val)
                else:
                    start_idx = int(np.clip(min_val, 0, n_points))
                    end_idx = int(np.clip(max_val, 0, n_points))
            except:
                start_idx = 0; end_idx = n_points
        else:
            start_idx = 0; end_idx = n_points

        start_idx = max(0, min(start_idx, n_points - 1))
        end_idx = max(0, min(end_idx, n_points))
        
        if end_idx <= start_idx: 
            if self.view_highlight: self.view_highlight.hide()
            return

        sub_coords = self.world_coords[start_idx:end_idx]
        
        if len(sub_coords) > 500:
            step = len(sub_coords) // 500
            sub_coords = sub_coords[::step]

        points = [QgsPointXY(x,y) for x,y in sub_coords]
        geom = QgsGeometry.fromPolylineXY(points)
        
        if self.view_highlight is None:
            self.view_highlight = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
            self.view_highlight.setColor(QColor(Qt.red))
            self.view_highlight.setWidth(4)
        
        # Wrap in try-except in case layer CRS access fails
        try:
            if self.qgis_layer.isValid():
                self.view_highlight.setToGeometry(geom, self.qgis_layer.crs())
                self.view_highlight.show()
        except:
            pass

    # =========================================================================
    # --- STATE PERSISTENCE ---
    # =========================================================================

    def get_state(self):
        """Returns a dict of current state for QGIS project storage."""
        if not self.data_manager: return None
        state = {
            "file_path": self.data_manager.file_path,
            "x_min": self.view.spin_x_min.value(),
            "x_max": self.view.spin_x_max.value(),
            "y_min": self.view.spin_y_min.value(),
            "y_max": self.view.spin_y_max.value(),
            "header": self.view.combo_header.currentText(),
            "contrast": self.view.spin_contrast.value(),
            "cmap": self.view.combo_cmap.currentText(),
            "domain": self.view.combo_domain.currentText(),
            "flip_x": self.view.chk_flip_x.isChecked(),
            "grid": self.view.chk_grid.isChecked(),
            "manual_step": self.view.chk_manual_step.isChecked(),
            "step_val": self.view.spin_step.value()
        }
        return state

    def restore_state(self, state):
        """Restores state from dict and rebuilds navigation index."""
        if not state: return
        
        path = state.get("file_path", "")
        if not path or not os.path.exists(path):
            self.view.update_status(f"File not found: {path}"); return

        try:
            self.view.update_status("Restoring session...")
            
            # 1. Initialize Data Manager if not already set
            if not self.data_manager:
                self.data_manager = SeismicDataManager(path)
            
            self.full_cum_dist = None; self.dist_unit = "m"
            
            # Re-enable UI Elements
            self.action_text_header.setEnabled(True); self.action_header_qc.setEnabled(True)
            self.action_agc.setEnabled(True); self.action_filter.setEnabled(True)
            self.action_reset.setEnabled(True); self.action_spectrum.setEnabled(True)
            self.action_dist.setEnabled(True); self.action_histogram.setEnabled(True)
            self.act_env.setEnabled(True); self.act_phase.setEnabled(True)
            self.act_cos.setEnabled(True); self.act_freq.setEnabled(True)
            self.act_rms.setEnabled(True)
            
            self.view.chk_manual_step.setChecked(False)
            self.view.spin_step.setEnabled(False)
            self.view.combo_header.clear(); self.view.combo_header.addItem("Trace Index")
            self.view.combo_header.addItems(self.data_manager.available_headers)

            # Restore settings
            if "header" in state: self.view.combo_header.setCurrentText(state["header"])
            self.on_header_changed() 

            if "cmap" in state: self.view.combo_cmap.setCurrentText(state["cmap"])
            if "contrast" in state: self.view.spin_contrast.setValue(float(state["contrast"]))
            if "domain" in state: self.view.combo_domain.setCurrentText(state["domain"])
            if "flip_x" in state: self.view.chk_flip_x.setChecked(state["flip_x"])
            if "grid" in state: self.view.chk_grid.setChecked(state["grid"])
            
            if state.get("manual_step", False):
                self.view.chk_manual_step.setChecked(True)
                self.view.spin_step.setValue(int(state.get("step_val", 1)))

            self.apply_changes()
            
            x_min = float(state.get("x_min", 0))
            x_max = float(state.get("x_max", self.data_manager.n_traces))
            
            if x_max <= x_min or (x_max - x_min) < 10:
                x_min = 0; x_max = self.data_manager.n_traces
                
            self.view.spin_x_min.setValue(x_min)
            self.view.spin_x_max.setValue(x_max)
            if "y_min" in state: self.view.spin_y_min.setValue(float(state["y_min"]))
            if "y_max" in state: self.view.spin_y_max.setValue(float(state["y_max"]))
            
            # --- Rebuild Navigation Index from Layer Metadata ---
            if self.qgis_layer and self.qgis_layer.isValid():
                # Defaults
                x_key = "CDP_X"; y_key = "CDP_Y"; use_header = True
                scalar_key = "Source_Group_Scalar"; manual_val = 1.0
                
                # Load saved geometry params
                try:
                    p_json = self.qgis_layer.customProperty("seisplotpy_geometry_params")
                    if p_json:
                        p = json.loads(str(p_json))
                        x_key = p.get("x_key", x_key)
                        y_key = p.get("y_key", y_key)
                        use_header = p.get("use_header", use_header)
                        scalar_key = p.get("scalar_key", scalar_key)
                        manual_val = float(p.get("manual_val", manual_val))
                except Exception: pass

                # Build the Tree
                try:
                    self.coord_step = 10 
                    raw_x = self.data_manager.get_header_slice(x_key, 0, self.data_manager.n_traces, self.coord_step)
                    raw_y = self.data_manager.get_header_slice(y_key, 0, self.data_manager.n_traces, self.coord_step)
                    
                    if use_header:
                        scalars = self.data_manager.get_header_slice(scalar_key, 0, self.data_manager.n_traces, self.coord_step)
                        cdp_x = SeismicProcessing.apply_scalar(raw_x, scalars)
                        cdp_y = SeismicProcessing.apply_scalar(raw_y, scalars)
                    else:
                        cdp_x = raw_x * manual_val
                        cdp_y = raw_y * manual_val
                    
                    if np.any(cdp_x) and np.any(cdp_y):
                        self.world_coords = np.column_stack((cdp_x, cdp_y))
                        self.coord_tree = cKDTree(self.world_coords)
                        self.view.update_status(f"Navigation Index Rebuilt: {len(cdp_x)} points")
                except Exception as e:
                    print(f"SeisPlotPy: Error calculating coordinates: {e}")

            elif not self.qgis_layer:
                # Fallback if no layer exists
                try:
                    cdp_x = self.data_manager.get_header_slice('CDP_X', 0, self.data_manager.n_traces, 10)
                    cdp_y = self.data_manager.get_header_slice('CDP_Y', 0, self.data_manager.n_traces, 10)
                    if np.any(cdp_x) and np.any(cdp_y):
                         self.create_qgis_layer(cdp_x, cdp_y, None) 
                except: pass
            
            self.load_horizons()
            
            self.view.btn_load.setEnabled(False)
            self.view.btn_load.setText(f"Linked: {os.path.basename(path)}")
            
        except Exception as e:
            self.view.update_status(f"Restore failed: {e}")
            print(f"SeisPlotPy Restore Error: {e}")

    def save_horizons(self):
        """Saves horizons to sidecar JSON."""
        if not self.data_manager: return
        path = self.data_manager.file_path + ".horizons.json"
        try:
            data = self.horizon_manager.get_state()
            with open(path, 'w') as f:
                json.dump(data, f)
        except: pass

    def load_horizons(self):
        """Loads horizons from sidecar JSON."""
        if not self.data_manager: return
        path = self.data_manager.file_path + ".horizons.json"
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.horizon_manager.restore_state(data)
                self.draw_horizons()
            except: pass

    # =========================================================================
    # --- CORE & UI LOGIC ---
    # =========================================================================

    def on_plot_clicked(self, event):
        if not self.is_picking_mode: return
        pos = event.scenePos()
        if self.view.plot_widget.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.view.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            click_x = mousePoint.x()
            click_y = mousePoint.y()
            trace_idx = int(click_x)
            header = self.view.combo_header.currentText()
            if header == "Trace Index":
                trace_idx = int(round(click_x))
            elif self.active_header_map is not None:
                if len(self.active_header_map) > 0:
                    trace_idx = int((np.abs(self.active_header_map - click_x)).argmin())
            if event.button() == Qt.LeftButton:
                self.horizon_manager.add_point(trace_idx, click_y)
            elif event.button() == Qt.RightButton:
                view_range = self.view.plot_widget.viewRange()
                y_height = abs(view_range[1][1] - view_range[1][0])
                y_tol = y_height * 0.02 
                self.horizon_manager.delete_closest_point(trace_idx, click_y, tolerance_x=5, tolerance_y=y_tol)

    def set_picking_mode(self, active, horizon_name):
        self.is_picking_mode = active
        self.view.plot_widget.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def reset_view(self):
        if not self.data_manager: return
        header = self.view.combo_header.currentText()
        x_min, x_max = 0, 0
        if header == "Trace Index":
            x_min, x_max = 0, self.data_manager.n_traces
        elif header == "Cumulative Distance" and self.full_cum_dist is not None:
            x_min, x_max = self.full_cum_dist[0], self.full_cum_dist[-1]
        elif self.active_header_map is not None:
            x_min, x_max = np.min(self.active_header_map), np.max(self.active_header_map)
        t_min, t_max = self.data_manager.time_axis[0], self.data_manager.time_axis[-1]
        self.view.spin_x_min.setValue(x_min)
        self.view.spin_x_max.setValue(x_max)
        self.view.spin_y_min.setValue(t_min)
        self.view.spin_y_max.setValue(t_max)
        self.view.chk_manual_step.setChecked(False)
        self.apply_changes()

    def calculate_distance(self, settings):
        print("Calculating Cumulative Distance...")
        try:
            raw_x = self.data_manager.get_header_slice(settings['x_key'], 0, self.data_manager.n_traces, 1)
            raw_y = self.data_manager.get_header_slice(settings['y_key'], 0, self.data_manager.n_traces, 1)
            if settings['use_header']:
                scalars = self.data_manager.get_header_slice(settings['scalar_key'], 0, self.data_manager.n_traces, 1)
                scaled_x = SeismicProcessing.apply_scalar(raw_x, scalars)
                scaled_y = SeismicProcessing.apply_scalar(raw_y, scalars)
            else:
                s = settings['manual_val']
                scaled_x = raw_x * s
                scaled_y = raw_y * s
            
            # 1. Ask User for CRS
            crs = None
            selector = QgsProjectionSelectionDialog(self.view)
            selector.setMessage("Select CRS for Coordinates (e.g., UTM Zone)")
            if selector.exec():
                crs = selector.crs()
            
            # 2. Calculate Distance
            dist = SeismicProcessing.calculate_cumulative_distance(scaled_x, scaled_y)
            max_dist = dist[-1]
            if max_dist > 10000:
                dist = dist / 1000.0
                self.dist_unit = "km"
            else:
                self.dist_unit = "m"
            self.full_cum_dist = dist
            
            # 3. Create Layer with CRS
            self.create_qgis_layer(scaled_x, scaled_y, crs, settings)

            if self.view.combo_header.findText("Cumulative Distance") == -1:
                self.view.combo_header.insertItem(1, "Cumulative Distance")
            self.view.combo_header.setCurrentText("Cumulative Distance")
            self.on_header_changed()
            QMessageBox.information(self.view, "Success", f"Calculated distance. Length: {self.full_cum_dist[-1]:.2f} {self.dist_unit}")
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Calculation failed:\n{str(e)}")

    def update_labels(self):
        x_label = self.get_x_label()
        y_domain = self.view.combo_domain.currentText()
        self.view.update_labels(x_label, y_domain)

    def get_x_label(self):
        label = self.view.combo_header.currentText()
        if label == "Cumulative Distance": return f"Cumulative Distance ({self.dist_unit})"
        return label

    def get_scaled_header(self, header_name, start, end, step):
        if header_name == "Cumulative Distance":
            if self.full_cum_dist is not None: return self.full_cum_dist[start:end:step]
            else: return np.arange(start, end, step)
        raw = self.data_manager.get_header_slice(header_name, start, end, step)
        coord_keys = ['SourceX', 'SourceY', 'GroupX', 'GroupY', 'CDP_X', 'CDP_Y']
        if header_name in coord_keys and 'SourceGroupScalar' in self.data_manager.available_headers:
            scalars = self.data_manager.get_header_slice('SourceGroupScalar', start, end, step)
            return SeismicProcessing.apply_scalar(raw, scalars)
        return raw

    def show_dist_tool(self):
        if not self.data_manager: return
        dlg = GeometryDialog(self.data_manager.available_headers, self.view)
        if dlg.exec() == QDialog.Accepted:
            settings = dlg.get_settings()
            self.calculate_distance(settings)

    def load_file(self):
        # Lock: If we already have data, prevent overwriting in this window
        if self.data_manager is not None:
            QMessageBox.information(self.view, "File Already Loaded", 
                                    "This window is already linked to a seismic line.\n"
                                    "To load another line, please click the plugin button in the QGIS toolbar to open a new window.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self.view, "Open SEG-Y File", "", "SEG-Y Files (*.sgy *.segy)")
        if not file_path: return
        
        self.view.update_status("Loading... Please wait")
        QApplication.processEvents()
        
        try:
            self.data_manager = SeismicDataManager(file_path)
            self.full_cum_dist = None
            self.dist_unit = "m"
            
            self.action_text_header.setEnabled(True); self.action_header_qc.setEnabled(True)
            self.action_agc.setEnabled(True); self.action_filter.setEnabled(True); self.action_reset.setEnabled(True)
            self.action_spectrum.setEnabled(True); self.action_dist.setEnabled(True)
            self.act_env.setEnabled(True); self.act_phase.setEnabled(True); self.act_cos.setEnabled(True)
            self.act_freq.setEnabled(True); self.act_rms.setEnabled(True)
            
            self.view.chk_manual_step.setChecked(False); self.view.spin_step.setEnabled(False)
            self.view.combo_header.clear(); self.view.combo_header.addItem("Trace Index")
            self.view.combo_header.addItems(self.data_manager.available_headers)
            
            if "CDP" in self.data_manager.available_headers:
                self.view.combo_header.setCurrentText("CDP")
                self.active_header_map = self.data_manager.get_header_slice("CDP", 0, self.data_manager.n_traces, 1)
            else:
                self.view.combo_header.setCurrentText("Trace Index"); self.active_header_map = None
            
            total_traces = self.data_manager.n_traces
            smart_step = max(1, int(total_traces / 2000))
            
            self.load_data_internal(0, total_traces, smart_step, auto_fit=True)
            self.view.update_status(f"Loaded: {file_path.split('/')[-1]}\nTraces: {total_traces}")

            # Lock the load button
            self.view.btn_load.setEnabled(False)
            self.view.btn_load.setText(f"Linked: {os.path.basename(file_path)}")
            self.view.btn_load.setToolTip("File loaded. To open another line, use the QGIS Toolbar button.")
            
            if hasattr(self, 'action_load_menu'):
                self.action_load_menu.setEnabled(False)
                
            self.action_histogram.setEnabled(True)
            
        except Exception as e:
            self.view.update_status("Load failed.")
            QMessageBox.critical(self.view, "Error", f"Failed to load file:\n{str(e)}")

    def load_file_from_path(self, path, load_data_only=False):
        """Helper to load a file directly (internal use)."""
        if not path or not os.path.exists(path): return
        self.view.update_status("Loading... Please wait")
        QApplication.processEvents()
        
        try:
            self.data_manager = SeismicDataManager(path)
            self.full_cum_dist = None; self.dist_unit = "m"
            
            self.action_text_header.setEnabled(True); self.action_header_qc.setEnabled(True)
            self.action_agc.setEnabled(True); self.action_filter.setEnabled(True); self.action_reset.setEnabled(True)
            self.action_spectrum.setEnabled(True); self.action_dist.setEnabled(True)
            self.act_env.setEnabled(True); self.act_phase.setEnabled(True); self.act_cos.setEnabled(True)
            self.act_freq.setEnabled(True); self.act_rms.setEnabled(True)
            self.view.chk_manual_step.setChecked(False); self.view.spin_step.setEnabled(False)
            self.view.combo_header.clear(); self.view.combo_header.addItem("Trace Index")
            self.view.combo_header.addItems(self.data_manager.available_headers)
            if "CDP" in self.data_manager.available_headers:
                self.view.combo_header.setCurrentText("CDP")
                self.active_header_map = self.data_manager.get_header_slice("CDP", 0, self.data_manager.n_traces, 1)
            else:
                self.view.combo_header.setCurrentText("Trace Index"); self.active_header_map = None
            total_traces = self.data_manager.n_traces
            smart_step = max(1, int(total_traces / 2000))
            
            # Pass load_data_only flag down
            self.load_data_internal(0, total_traces, smart_step, auto_fit=True, create_layer=not load_data_only)
            
            self.view.update_status(f"Loaded: {path.split('/')[-1]}\nTraces: {total_traces}")
            self.view.btn_load.setEnabled(False)
            self.view.btn_load.setText(f"Linked: {os.path.basename(path)}")
            if hasattr(self, 'action_load_menu'): self.action_load_menu.setEnabled(False)
            self.action_histogram.setEnabled(True)
        except Exception as e:
            self.view.update_status("Load failed.")
            print(f"Load failed: {e}")


    def handle_horizon_export(self, index):
        if not self.data_manager: QMessageBox.warning(self.view, "Error", "No seismic data loaded."); return
        h = self.horizon_manager.horizons[index]; points = h['points']
        if not points: QMessageBox.warning(self.view, "Error", "Horizon is empty."); return
        dlg = HeaderExportDialog(self.data_manager.available_headers, self.view)
        if dlg.exec() != QDialog.Accepted: return
        selected_headers = dlg.get_selected_headers()
        path, _ = QFileDialog.getSaveFileName(self.view, "Save Horizon CSV", f"{h['name']}.csv", "CSV (*.csv)")
        if not path: return
        try:
            x_vals, y_vals = zip(*points); x_arr = np.array(x_vals); y_arr = np.array(y_vals)
            df = pd.DataFrame()
            current_x_mode = self.view.combo_header.currentText()
            current_y_mode = self.view.combo_domain.currentText()
            df[current_x_mode] = x_arr; df[current_y_mode] = y_arr
            trace_indices = np.zeros(len(x_arr), dtype=int)
            if current_x_mode == "Trace Index": trace_indices = np.round(x_arr).astype(int)
            elif self.active_header_map is not None:
                if np.all(np.diff(self.active_header_map) >= 0):
                    trace_indices = np.searchsorted(self.active_header_map, x_arr); trace_indices = np.clip(trace_indices, 0, len(self.active_header_map)-1)
                else:
                    for i, val in enumerate(x_arr): trace_indices[i] = (np.abs(self.active_header_map - val)).argmin()
            for hdr in selected_headers:
                full_hdr_vals = self.data_manager.get_header_slice(hdr, 0, self.data_manager.n_traces, 1)
                df[hdr] = full_hdr_vals[trace_indices]
            df.to_csv(path, index=False)
            QMessageBox.information(self.view, "Success", f"Saved horizon with {len(selected_headers)} extra headers.")
        except Exception as e: QMessageBox.critical(self.view, "Export Error", str(e))

    def on_header_changed(self):
        if not self.data_manager: return
        header = self.view.combo_header.currentText()
        if header == "Trace Index": self.active_header_map = None
        elif header == "Cumulative Distance": self.active_header_map = self.full_cum_dist
        else: self.active_header_map = self.get_scaled_header(header, 0, self.data_manager.n_traces, 1)
        start = self.loaded_start_trace; end = self.loaded_end_trace
        num_traces = self.current_data.shape[1]
        step = max(1, int((end-start)/num_traces)) if num_traces > 0 else 1
        if header == "Trace Index": self.x_vals = np.arange(start, end, step)
        else: self.x_vals = self.get_scaled_header(header, start, end, step)
        x_min = self.x_vals[0] if self.x_vals.size > 0 else 0; x_max = self.x_vals[-1] if self.x_vals.size > 0 else 1
        y_min = self.view.spin_y_min.value(); y_max = self.view.spin_y_max.value()
        self.view.display_seismic(self.current_data.T, x_range=(x_min, x_max), y_range=(y_min, y_max))
        self.is_programmatic_update = True
        self.view.spin_x_min.setValue(x_min); self.view.spin_x_max.setValue(x_max)
        self.is_programmatic_update = False
        self.update_labels(); self.draw_horizons()

    def apply_changes(self):
        if not self.data_manager: return
        self.view.update_status("Reloading data... Please wait"); QApplication.processEvents()
        target_x_min = self.view.spin_x_min.value(); target_x_max = self.view.spin_x_max.value()
        start_trace = int(target_x_min); end_trace = int(target_x_max)
        header = self.view.combo_header.currentText()
        if header != "Trace Index" and self.active_header_map is not None:
            if np.all(np.diff(self.active_header_map) >= 0):
                start_trace = np.searchsorted(self.active_header_map, target_x_min)
                end_trace = np.searchsorted(self.active_header_map, target_x_max)
            else:
                mask = (self.active_header_map >= min(target_x_min, target_x_max)) & (self.active_header_map <= max(target_x_min, target_x_max))
                indices = np.where(mask)[0]
                if indices.size > 0: start_trace = indices[0]; end_trace = indices[-1] + 1
                else: QMessageBox.warning(self.view, "Warning", f"No data found for range."); return
        start_trace = max(0, start_trace); end_trace = min(self.data_manager.n_traces, end_trace)
        if self.view.chk_manual_step.isChecked(): step = int(self.view.spin_step.value())
        else:
            trace_count = abs(end_trace - start_trace); step = max(1, int(trace_count / 2000))
            if trace_count < 5000: step = 1
        print(f"Reloading: Indices {start_trace}-{end_trace}, Step={step}")
        self.load_data_internal(start_trace, end_trace, step, auto_fit=False)
        self.is_programmatic_update = True
        self.view.plot_widget.setXRange(target_x_min, target_x_max, padding=0)
        self.view.plot_widget.setYRange(self.view.spin_y_min.value(), self.view.spin_y_max.value(), padding=0)
        self.view.plot_widget.setYRange(self.view.spin_y_min.value(), self.view.spin_y_max.value(), padding=0)
        self.is_programmatic_update = False
        self.update_layer_state() # Sync to layer
        self.view.update_status(f"Loaded: {self.data_manager.file_path.split('/')[-1]} (Subset)")

    def load_data_internal(self, start, end, step, auto_fit=False, create_layer=True):
        try:
            self.loaded_start_trace = max(0, start); self.loaded_end_trace = min(self.data_manager.n_traces, end)
            if self.loaded_start_trace >= self.loaded_end_trace: return
            self.current_data = self.data_manager.get_data_slice(self.loaded_start_trace, self.loaded_end_trace, step)
            header = self.view.combo_header.currentText()
            if header == "Trace Index": self.x_vals = np.arange(self.loaded_start_trace, self.loaded_end_trace, step)
            else: self.x_vals = self.get_scaled_header(header, self.loaded_start_trace, self.loaded_end_trace, step)
            self.t_vals = self.data_manager.time_axis
            x_min = self.x_vals[0] if self.x_vals.size > 0 else 0; x_max = self.x_vals[-1] if self.x_vals.size > 0 else 1
            t_min = self.t_vals[0]; t_max = self.t_vals[-1]
            self.is_programmatic_update = True
            self.view.display_seismic(self.current_data.T, x_range=(x_min, x_max), y_range=(t_min, t_max))
            if not self.view.chk_manual_step.isChecked(): self.view.spin_step.setValue(step)
            if auto_fit:
                self.view.plot_widget.autoRange(); self.view.spin_x_min.setValue(x_min); self.view.spin_x_max.setValue(x_max); self.view.spin_y_min.setValue(t_min); self.view.spin_y_max.setValue(t_max)
            self.update_contrast(); self.update_labels(); self.draw_horizons(); self.is_programmatic_update = False
        except Exception as e: print(f"Load error: {e}"); self.is_programmatic_update = False

    def sync_view_to_controls(self, _, ranges):
        if self.is_programmatic_update: return
        x_range, y_range = ranges
        self.view.spin_x_min.blockSignals(True); self.view.spin_x_max.blockSignals(True); self.view.spin_y_min.blockSignals(True); self.view.spin_y_max.blockSignals(True)
        self.view.spin_x_min.setValue(x_range[0]); self.view.spin_x_max.setValue(x_range[1]); self.view.spin_y_min.setValue(y_range[0]); self.view.spin_y_max.setValue(y_range[1])
        if not self.view.chk_manual_step.isChecked(): visible_width = abs(x_range[1] - x_range[0]); self.view.spin_step.setValue(max(1, int(visible_width / 2000)))
        
        self.highlight_map_extent(x_range)

        self.view.spin_x_min.blockSignals(False); self.view.spin_x_max.blockSignals(False); self.view.spin_y_min.blockSignals(False); self.view.spin_y_max.blockSignals(False)
    def toggle_manual_step(self, state): self.view.spin_step.setEnabled(state == 2)
    def toggle_flip_x(self, state): self.view.plot_widget.getPlotItem().invertX(state == 2)
    def toggle_grid(self, state): self.view.plot_widget.showGrid(x=(state == 2), y=(state == 2))
    
    def toggle_smooth(self, state):
        """Enables Bilinear Interpolation (Smooth Pixmap Transform)."""
        # State 2 means Checked
        self.view.plot_widget.setRenderHint(QPainter.SmoothPixmapTransform, state == 2)
        # We also need to trigger an update to see changes immediately
        self.view.plot_widget.update()

    def change_colormap(self, text): self.view.set_colormap(text)
    def draw_horizons(self):
        for item in self.horizon_items: self.view.plot_widget.removeItem(item)
        self.horizon_items = []
        header = self.view.combo_header.currentText()
        if header == "Trace Index": map_array = None
        elif header == "Cumulative Distance": map_array = self.full_cum_dist
        elif self.active_header_map is not None: map_array = self.active_header_map
        else: map_array = None
        for h in self.horizon_manager.horizons:
            if not h['visible'] or not h['points']: continue
            idx_data, y_data = zip(*h['points']); idx_arr = np.array(idx_data, dtype=int); y_arr = np.array(y_data)
            if map_array is not None: idx_arr = np.clip(idx_arr, 0, len(map_array)-1); x_arr = map_array[idx_arr]
            else: x_arr = idx_arr
            curve = pg.PlotCurveItem(x=x_arr, y=y_arr, pen=pg.mkPen(color=h['color'], width=2)); self.view.plot_widget.addItem(curve); self.horizon_items.append(curve)
            scatter = pg.ScatterPlotItem(x=x_arr, y=y_arr, size=5, brush=h['color'], pen=None); self.view.plot_widget.addItem(scatter); self.horizon_items.append(scatter)
    def setup_menu(self):
        menu_bar = self.view.menuBar()
        file_menu = menu_bar.addMenu("File")
        self.action_load_menu = file_menu.addAction("Load SEG-Y", self.load_file)
        file_menu.addAction("Export PDF/PNG", self.export_figure)
        proc_menu = menu_bar.addMenu("Processing"); self.action_agc = QAction("Apply AGC", self.view); self.action_agc.triggered.connect(self.run_agc); self.action_agc.setEnabled(False); proc_menu.addAction(self.action_agc); self.action_filter = QAction("Bandpass Filter", self.view); self.action_filter.triggered.connect(self.run_filter); self.action_filter.setEnabled(False); proc_menu.addAction(self.action_filter); proc_menu.addSeparator(); self.action_reset = QAction("Reset to Raw Data", self.view); self.action_reset.triggered.connect(self.reset_processing); self.action_reset.setEnabled(False); proc_menu.addAction(self.action_reset)
        attr_menu = menu_bar.addMenu("Attributes"); self.act_env = QAction("Instantaneous Amplitude (Envelope)", self.view); self.act_env.triggered.connect(lambda: self.run_attribute("Envelope")); self.act_env.setEnabled(False); attr_menu.addAction(self.act_env); self.act_phase = QAction("Instantaneous Phase", self.view); self.act_phase.triggered.connect(lambda: self.run_attribute("Phase")); self.act_phase.setEnabled(False); attr_menu.addAction(self.act_phase); self.act_cos = QAction("Cosine of Phase", self.view); self.act_cos.triggered.connect(lambda: self.run_attribute("Cosine Phase")); self.act_cos.setEnabled(False); attr_menu.addAction(self.act_cos); self.act_freq = QAction("Instantaneous Frequency", self.view); self.act_freq.triggered.connect(lambda: self.run_attribute("Frequency")); self.act_freq.setEnabled(False); attr_menu.addAction(self.act_freq); attr_menu.addSeparator(); self.act_rms = QAction("RMS Amplitude", self.view); self.act_rms.triggered.connect(lambda: self.run_attribute("RMS")); self.act_rms.setEnabled(False); attr_menu.addAction(self.act_rms)
        tools_menu = menu_bar.addMenu("Tools"); self.action_dist = QAction("Setup Geometry / Distance", self.view); self.action_dist.triggered.connect(self.show_dist_tool); self.action_dist.setEnabled(False); tools_menu.addAction(self.action_dist); self.action_horizons = QAction("Horizon Manager & Picking", self.view); self.action_horizons.triggered.connect(lambda: self.horizon_manager.show()); tools_menu.addAction(self.action_horizons); self.action_text_header = QAction("View Text Header", self.view); self.action_text_header.triggered.connect(self.show_text_header); self.action_text_header.setEnabled(False); tools_menu.addAction(self.action_text_header); self.action_header_qc = QAction("Trace Header QC Plot", self.view); self.action_header_qc.triggered.connect(self.show_header_qc); self.action_header_qc.setEnabled(False); tools_menu.addAction(self.action_header_qc); self.action_spectrum = QAction("Frequency Spectrum", self.view); self.action_spectrum.triggered.connect(self.show_spectrum); self.action_spectrum.setEnabled(False); tools_menu.addAction(self.action_spectrum)
        self.action_histogram = QAction("View Amplitude Histogram", self.view)
        self.action_histogram.triggered.connect(self.show_amplitude_histogram)
        self.action_histogram.setEnabled(False)
        tools_menu.addAction(self.action_histogram)
    def show_text_header(self): 
        if self.data_manager: TextHeaderDialog(self.data_manager.get_text_header(), self.view).exec()
    def show_header_qc(self): 
        if self.data_manager: HeaderQCPlot(self.data_manager.available_headers, self.data_manager, self.view).exec()
    def show_spectrum(self):
        if self.current_data is None: return
        try: freqs, amps = SeismicProcessing.calculate_spectrum(self.current_data, self.data_manager.sample_rate); self.spectrum_dlg = SpectrumPlot(freqs, amps, self.view); self.spectrum_dlg.show()
        except Exception as e: QMessageBox.critical(self.view, "Error", str(e))
    def run_agc(self):
        if self.current_data is None: return
        window, ok = QInputDialog.getInt(self.view, "AGC Settings", "Window Size (ms):", 500, 10, 5000)
        if not ok: return
        try: self.current_data = SeismicProcessing.apply_agc(self.current_data, self.data_manager.sample_rate, window); self.update_display_only(); self.view.update_status("Applied AGC")
        except Exception as e: QMessageBox.critical(self.view, "Processing Error", str(e))
    def run_filter(self):
        if self.current_data is None: return
        dlg = BandpassDialog(self.view)
        if dlg.exec():
            low, high = dlg.get_values()
            try: self.current_data = SeismicProcessing.apply_bandpass(self.current_data, self.data_manager.sample_rate, low, high); self.update_display_only(); self.view.update_status(f"Applied Bandpass {low}-{high} Hz")
            except Exception as e: QMessageBox.critical(self.view, "Processing Error", str(e))
    def reset_processing(self): self.apply_changes(); self.view.update_status("Reset to Raw Data")
    
    # --- UPDATED: Fixes the Squeezed Display Bug ---
    def update_display_only(self):
        if self.current_data is None: return
        
        # 1. Determine correct X bounds from data, NOT spinboxes
        # The spinboxes contain the VIEW limits (e.g. 1000-2000), 
        # but the data might be traces 1002-1998 (due to integer snapping)
        if hasattr(self, 'x_vals') and self.x_vals is not None and self.x_vals.size > 0:
            x_min = self.x_vals[0]
            x_max = self.x_vals[-1]
        else:
            x_min = self.view.spin_x_min.value()
            x_max = self.view.spin_x_max.value()
            
        # 2. Determine correct Y bounds from data
        # Data is always full time range, so we must use the full time range for the Image Rect
        # otherwise we squash 5 seconds of data into a 1 second view box
        if hasattr(self, 't_vals') and self.t_vals is not None:
            y_min = self.t_vals[0]
            y_max = self.t_vals[-1]
        else:
            y_min = self.view.spin_y_min.value()
            y_max = self.view.spin_y_max.value()
        
        # 3. Display with DATA bounds
        self.view.display_seismic(self.current_data.T, x_range=(x_min, x_max), y_range=(y_min, y_max))
        self.update_contrast(); self.draw_horizons()
    # -----------------------------------------------

    def match_aspect_ratio(self):
        try:
            target_w = self.view.spin_fig_width.value(); target_h = self.view.spin_fig_height.value(); 
            if target_h == 0: return
            target_ratio = target_w / target_h
            current_plot_h = self.view.plot_widget.height(); new_plot_w = int(current_plot_h * target_ratio); current_plot_w = self.view.plot_widget.width(); diff_w = new_plot_w - current_plot_w
            self.view.resize(self.view.width() + diff_w, self.view.height())
        except Exception: pass
    def update_contrast(self):
        if self.current_data is None: return
        try:
            p = self.view.spin_contrast.value()
            if self.current_data.size > 0: clip_val = np.percentile(np.abs(self.current_data), p)
            else: clip_val = 1.0
            self.view.img_item.setImage(self.current_data.T, levels=[-clip_val, clip_val], autoLevels=False)
        except Exception: pass
    def export_figure(self):
        if self.current_data is None: QMessageBox.warning(self.view, "Warning", "No data to export."); return
        dpi, ok = QInputDialog.getInt(self.view, "Export Settings", "DPI (Resolution):", 300, 72, 1200)
        if not ok: return
        file_path, _ = QFileDialog.getSaveFileName(self.view, "Save Figure", "seismic_plot.pdf", "PDF Documents (*.pdf);;PNG Images (*.png)")
        if not file_path: return
        try:
            w = self.view.spin_fig_width.value(); h = self.view.spin_fig_height.value()
            fig, ax = plt.subplots(figsize=(w, h))
            p = self.view.spin_contrast.value()
            clip_val = np.percentile(np.abs(self.current_data), p) if self.current_data.size > 0 else 1.0
            extent = [self.x_vals[0], self.x_vals[-1], self.t_vals[-1], self.t_vals[0]]
            im = ax.imshow(self.current_data, cmap=self.view.combo_cmap.currentText(), aspect='auto', extent=extent, vmin=-clip_val, vmax=clip_val, interpolation='lanczos')
            if self.view.chk_grid.isChecked():
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                ax.grid(False)
            ax.set_xlabel(self.get_x_label()) 
            ylabel = "TWT (ms)" if self.view.combo_domain.currentText() == "Time" else "Depth (m)"; ax.set_ylabel(ylabel)
            ax.set_xlim(self.view.spin_x_min.value(), self.view.spin_x_max.value())
            ax.set_ylim(self.view.spin_y_max.value(), self.view.spin_y_min.value())
            if self.view.chk_flip_x.isChecked(): ax.invert_xaxis()
            for h in self.horizon_manager.horizons:
                if h['visible'] and h['points']:
                    x_d, y_d = zip(*h['points'])
                    ax.plot(x_d, y_d, color=h['color'], linewidth=1.0)
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', metadata={'Creator': 'SeisPlotPy'})
            plt.close(fig); QMessageBox.information(self.view, "Success", f"Exported to:\n{file_path}")
        except Exception as e: QMessageBox.critical(self.view, "Export Failed", str(e))
    def run_attribute(self, attr_type):
        if self.data_manager is not None:
            # 1. Get current view range from the plot (what the user is looking at)
            view_range = self.view.plot_widget.viewRange()
            x_min, x_max = view_range[0]
            
            # 2. Convert coordinates back to trace indices
            start_trace, end_trace = 0, 0
            header = self.view.combo_header.currentText()
            
            if header == "Trace Index":
                start_trace = int(x_min)
                end_trace = int(x_max)
            elif self.active_header_map is not None:
                # Use searchsorted for fast lookup if headers are mapped
                try:
                    # Sort check is needed because headers might be descending
                    if self.active_header_map[0] < self.active_header_map[-1]:
                        start_trace = np.searchsorted(self.active_header_map, x_min)
                        end_trace = np.searchsorted(self.active_header_map, x_max)
                    else:
                        # Handle reverse sorted headers (common in some surveys)
                        start_trace = np.searchsorted(self.active_header_map[::-1], x_min)
                        end_trace = np.searchsorted(self.active_header_map[::-1], x_max)
                        # Flip indices because we searched the reversed array
                        n = len(self.active_header_map)
                        start_trace, end_trace = n - end_trace, n - start_trace
                except:
                    # Fallback to simple scaling if search fails
                    start_trace = max(0, int(x_min))
                    end_trace = min(self.data_manager.n_traces, int(x_max))

            # Clamp to file limits
            start_trace = max(0, start_trace)
            end_trace = min(self.data_manager.n_traces, end_trace)
            
            # 3. CRITICAL: Force reload at FULL RESOLUTION (step=1) for the attribute math
            # We don't want to calculate attributes on decimated data!
            self.view.update_status(f"Fetching high-res data for {attr_type}...")
            QApplication.processEvents()
            
            # Update internal state so the 'Apply' button knows where we are
            self.view.spin_x_min.setValue(x_min)
            self.view.spin_x_max.setValue(x_max)
            self.view.chk_manual_step.setChecked(True) 
            self.view.spin_step.setValue(1) # Force step 1
            
            # Perform the load
            self.load_data_internal(start_trace, end_trace, step=1, auto_fit=False)

        # 4. Now calculate the attribute on the fresh, high-res data
        if self.current_data is None: return
        
        self.view.update_status(f"Calculating {attr_type}...")
        QApplication.processEvents()
        
        try:
            sr = self.data_manager.sample_rate
            
            if attr_type == "Envelope":
                self.current_data = SeismicProcessing.attribute_envelope(self.current_data)
            elif attr_type == "Phase":
                self.current_data = SeismicProcessing.attribute_phase(self.current_data)
                self.view.combo_cmap.setCurrentText("seismic") # Phase looks best in divergent colormap
            elif attr_type == "Frequency":
                self.current_data = SeismicProcessing.attribute_frequency(self.current_data, sr)
            elif attr_type == "Cosine Phase":
                self.current_data = SeismicProcessing.attribute_cosine_phase(self.current_data)
            elif attr_type == "RMS":
                window, ok = QInputDialog.getInt(self.view, "RMS Settings", "Window (ms):", 100, 10, 1000)
                if not ok: return
                self.current_data = SeismicProcessing.attribute_rms(self.current_data, sr, window)
            
            # 5. Refresh display
            self.update_display_only()
            self.view.update_status(f"Displayed: {attr_type} (High Res)")
            
        except Exception as e:
            QMessageBox.critical(self.view, "Attribute Error", str(e))
            self.view.update_status("Error calculating attribute")
    
    def publish_horizon_to_map(self, index):
        """Creates a QGIS vector layer for the selected horizon."""
        if not self.data_manager or not self.qgis_layer:
            self.view.update_status("Error: No seismic navigation layer linked.")
            return

        horizon = self.horizon_manager.horizons[index]
        points = horizon['points']
        if not points:
            self.view.update_status("Error: Horizon is empty.")
            return

        name = horizon['name']
        color_hex = horizon['color']

        try:
            # 1. Retrieve Geometry Settings
            x_key = "CDP_X"; y_key = "CDP_Y"; use_header = True
            scalar_key = "Source_Group_Scalar"; manual_val = 1.0

            p_json = self.qgis_layer.customProperty("seisplotpy_geometry_params")
            if p_json:
                p = json.loads(str(p_json))
                x_key = p.get("x_key", x_key)
                y_key = p.get("y_key", y_key)
                use_header = p.get("use_header", use_header)
                scalar_key = p.get("scalar_key", scalar_key)
                manual_val = float(p.get("manual_val", manual_val))

            # 2. Extract Trace Indices
            sorted_pts = sorted(points, key=lambda k: k[0])
            trace_indices = np.array([int(p[0]) for p in sorted_pts])
            times = np.array([p[1] for p in sorted_pts])

            # 3. Fetch Coordinate Arrays (Full Line)
            raw_x = self.data_manager.get_header_slice(x_key, 0, self.data_manager.n_traces, 1)
            raw_y = self.data_manager.get_header_slice(y_key, 0, self.data_manager.n_traces, 1)
            
            if use_header:
                scalars = self.data_manager.get_header_slice(scalar_key, 0, self.data_manager.n_traces, 1)
                full_x = SeismicProcessing.apply_scalar(raw_x, scalars)
                full_y = SeismicProcessing.apply_scalar(raw_y, scalars)
            else:
                full_x = raw_x * manual_val
                full_y = raw_y * manual_val

            # 4. Map Horizon Indices to Coordinates
            trace_indices = np.clip(trace_indices, 0, len(full_x) - 1)
            mapped_x = full_x[trace_indices]
            mapped_y = full_y[trace_indices]

            # 5. Create Vector Layer
            layer_crs = self.qgis_layer.crs().authid()
            crs_def = f"?crs={layer_crs}" if layer_crs else ""
            
            vl = QgsVectorLayer(f"LineString{crs_def}", f"{name} (Horizon)", "memory")
            pr = vl.dataProvider()
            
            pr.addAttributes([QgsField("first_trace", QVariant.Int), 
                              QgsField("last_trace", QVariant.Int),
                              QgsField("avg_time", QVariant.Double)])
            vl.updateFields()

            # Build Geometry
            qgs_pts = [QgsPointXY(x, y) for x, y in zip(mapped_x, mapped_y)]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(qgs_pts))
            feat.setAttributes([int(trace_indices[0]), int(trace_indices[-1]), float(np.mean(times))])
            
            pr.addFeatures([feat])
            vl.updateExtents()
            
            # 6. Apply Style
            symbol = QgsSymbol.defaultSymbol(vl.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setWidth(0.8)
            vl.setRenderer(QgsSingleSymbolRenderer(symbol))
            
            # 7. Add to Project
            QgsProject.instance().addMapLayer(vl)
            
            # --- STATUS UPDATE ONLY (No Pop-up) ---
            self.view.update_status(f"Published horizon '{name}' to map.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.view, "Error", f"Failed to publish horizon: {e}")
    def show_amplitude_histogram(self):
        """Display amplitude distribution with percentile clip lines"""
        if self.current_data is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Get CURRENT percentile value from the UI spin box
            percentile = self.view.spin_contrast.value()
            
            # Flatten data
            amps = self.current_data.flatten()
            amps = amps[np.isfinite(amps)]
            
            # Calculate clip value based on CURRENT percentile setting
            clip_val = np.percentile(np.abs(amps), percentile)
            
            # Count samples in each region
            below_clip = np.sum(amps < -clip_val)
            within_clip = np.sum((amps >= -clip_val) & (amps <= clip_val))
            above_clip = np.sum(amps > clip_val)
            
            # Create figure with larger default size
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Calculate better bin count (fewer bins = smoother histogram)
            # FIX 1: Ensure int
            n_bins_full = min(100, int(np.sqrt(len(amps))))  
            
            # FIX 1: within_clip is a scalar (count), so len() fails. Use the scalar directly.
            n_bins_zoom = min(80, int(np.sqrt(within_clip)))
            
            # --- TOP PLOT: Full histogram (LOG SCALE) ---
            ax1.hist(amps, bins=n_bins_full, color='steelblue', alpha=0.7, 
                    edgecolor='black', linewidth=0.5, log=True)
            ax1.axvline(-clip_val, color='red', linestyle='--', linewidth=3, 
                    label=f'Clip threshold (P{percentile:.1f})', zorder=5)
            ax1.axvline(clip_val, color='red', linestyle='--', linewidth=3, zorder=5)
            
            # Shade saturated regions (color-agnostic labels)
            ax1.axvspan(amps.min(), -clip_val, alpha=0.15, color='blue', 
                    label='Will saturate (low)', zorder=1)
            ax1.axvspan(clip_val, amps.max(), alpha=0.15, color='red', 
                    label='Will saturate (high)', zorder=1)
            
            ax1.set_xlabel('Amplitude', fontsize=14)
            ax1.set_ylabel('Frequency (log scale)', fontsize=14)
            ax1.set_title(f'Amplitude Distribution - All {len(amps):,} samples shown', 
                        fontsize=16, fontweight='bold', pad=15)
            ax1.legend(loc='upper right', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Statistics box - LARGER
            stats_text = (
                f'Total samples: {len(amps):,}\n'
                f'Min: {amps.min():.2e}\n'
                f'Max: {amps.max():.2e}\n'
                f'Mean: {amps.mean():.2e}\n'
                f'Std: {amps.std():.2e}\n'
                f'{"─"*25}\n'
                f'Clip value: ±{clip_val:.2e}\n'
                f'Below clip: {below_clip:,} ({below_clip/len(amps)*100:.2f}%)\n'
                f'Within range: {within_clip:,} ({within_clip/len(amps)*100:.2f}%)\n'
                f'Above clip: {above_clip:,} ({above_clip/len(amps)*100:.2f}%)'
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
                    fontfamily='monospace', fontsize=11)  # Larger font
            
            # --- BOTTOM PLOT: Zoomed to display range ---
            clipped_amps = amps[(amps >= -clip_val) & (amps <= clip_val)]
            ax2.hist(clipped_amps, bins=n_bins_zoom, color='green', alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
            ax2.axvline(-clip_val, color='red', linestyle='--', linewidth=3, 
                    label='Display boundaries', zorder=5)
            ax2.axvline(clip_val, color='red', linestyle='--', linewidth=3, zorder=5)
            
            ax2.set_xlabel('Amplitude', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.set_title(f'Dynamic Range Used for Color Mapping\n'
                        f'{len(clipped_amps):,} samples ({within_clip/len(amps)*100:.2f}%) '
                        f'mapped to colorbar', fontsize=16, fontweight='bold', pad=15)
            ax2.legend(loc='upper right', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            
            # Option to save
            # FIX 2: Use qgis.PyQt.QtWidgets or PyQt5
            from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog
            result = QMessageBox.question(
                self.view, 
                "Save Histogram?",
                f"Current percentile: P{percentile:.1f}\n"
                f"Saturated samples: {(below_clip + above_clip)/len(amps)*100:.2f}%\n\n"
                f"Save figure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.Yes:
                file_path, _ = QFileDialog.getSaveFileName(
                    self.view, 
                    "Save Histogram", 
                    f"amplitude_histogram_P{percentile:.0f}.png",
                    "PNG Images (*.png);;PDF Documents (*.pdf)"
                )
                if file_path:
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self.view, "Success", f"Saved to:\n{file_path}")
            
            plt.show()
            
        except Exception as e:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.critical(self.view, "Histogram Error", str(e))
