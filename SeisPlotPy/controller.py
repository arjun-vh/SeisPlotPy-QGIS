import os
from qgis.PyQt.QtWidgets import (QApplication, QFileDialog, QMessageBox, QInputDialog, 
                             QMenuBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QDoubleSpinBox, QDialog, QDialogButtonBox, 
                             QComboBox, QRadioButton, QButtonGroup, QAction)
from qgis.PyQt.QtGui import QCursor, QIcon, QColor
from qgis.PyQt.QtCore import Qt, QVariant

# --- QGIS IMPORTS ---
from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, 
                       QgsPointXY, QgsProject, QgsField, 
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsWkbTypes)
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
        

        # --- Hook into the Window Close Event ---
        self.view.closeEvent = self.cleanup_on_close
        

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
        self.view.btn_preview_ratio.clicked.connect(self.match_aspect_ratio)
        self.view.plot_widget.sigRangeChanged.connect(self.sync_view_to_controls)
        self.view.combo_header.activated.connect(self.on_header_changed)
        
        self.horizon_manager.picking_toggled.connect(self.set_picking_mode)
        self.horizon_manager.horizon_visibility_changed.connect(self.draw_horizons)
        self.horizon_manager.horizon_color_changed.connect(self.draw_horizons)
        self.horizon_manager.horizon_removed.connect(self.draw_horizons)
        
        # This line was causing the error because the method was missing
        self.horizon_manager.export_requested.connect(self.handle_horizon_export)
        
        self.view.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        
        self.view.show()

    # =========================================================================
    # --- CLEANUP & MAP INTERACTION ---
    # =========================================================================
    def cleanup_on_close(self, event):
        """Called when the user closes the Seismic Window."""
        if self.view_highlight:
            self.view_highlight.reset(QgsWkbTypes.LineGeometry)
            self.view_highlight = None
        event.accept()

    def create_qgis_layer(self, x_coords, y_coords, crs=None):
        """Creates QGIS layer AND builds spatial index for navigation."""
        layer_name = os.path.basename(self.data_manager.file_path)
        
        self.world_coords = np.column_stack((x_coords, y_coords))
        self.coord_tree = cKDTree(self.world_coords)
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
        
        if self.iface.mapCanvas():
            self.iface.mapCanvas().setExtent(layer.extent())
            self.iface.mapCanvas().refresh()

    def _transform_mouse_point(self, point):
        """Helper to transform Map Canvas Point -> Layer CRS Point."""
        if self.qgis_layer is None: return point
        
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        layer_crs = self.qgis_layer.crs() 

        if canvas_crs != layer_crs and layer_crs.isValid():
            try:
                xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
                return xform.transform(point)
            except Exception:
                return point 
        return point

    def handle_map_hover(self, point):
        """Received from SeisPlotPy when mouse moves on canvas."""
        if self.coord_tree is None: return
        
        search_point = self._transform_mouse_point(point)
        dist, idx = self.coord_tree.query([search_point.x(), search_point.y()])

        mupp = self.iface.mapCanvas().mapUnitsPerPixel()
        tolerance = mupp * 20 
        
        if dist > tolerance: 
            if self.map_marker: self.map_marker.hide()
            return

        plot_x_value = 0
        current_header = self.view.combo_header.currentText()
        
        if current_header == "Trace Index":
            plot_x_value = idx
        elif self.active_header_map is not None and idx < len(self.active_header_map):
            plot_x_value = self.active_header_map[idx]
        else:
            plot_x_value = idx

        self.update_map_marker(plot_x_value)

    def handle_map_click(self, point):
        """Received from SeisPlotPy on double-click."""
        if self.coord_tree is None: return
        
        search_point = self._transform_mouse_point(point)
        dist, _ = self.coord_tree.query([search_point.x(), search_point.y()])
        
        mupp = self.iface.mapCanvas().mapUnitsPerPixel()
        tolerance = mupp * 10
        
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
        """Draws a red line on the QGIS map showing the zoomed area."""
        if self.coord_tree is None or self.world_coords is None: return
        if self.qgis_layer is None: return

        min_val, max_val = x_range
        start_idx, end_idx = 0, 0
        
        header = self.view.combo_header.currentText()
        n_points = self.world_coords.shape[0]

        if header == "Trace Index":
            start_idx = int(min_val)
            end_idx = int(max_val)
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
        
        self.view_highlight.setToGeometry(geom, self.qgis_layer.crs())
        self.view_highlight.show()

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
            self.create_qgis_layer(scaled_x, scaled_y, crs)

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
        self.is_programmatic_update = False
        self.view.update_status(f"Loaded: {self.data_manager.file_path.split('/')[-1]} (Subset)")

    def load_data_internal(self, start, end, step, auto_fit=False):
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
    
    
    def update_display_only(self):
        if self.current_data is None: return
        
        if hasattr(self, 'x_vals') and self.x_vals is not None and self.x_vals.size > 0:
            x_min = self.x_vals[0]
            x_max = self.x_vals[-1]
        else:
            x_min = self.view.spin_x_min.value()
            x_max = self.view.spin_x_max.value()
            
        if hasattr(self, 't_vals') and self.t_vals is not None:
            y_min = self.t_vals[0]
            y_max = self.t_vals[-1]
        else:
            y_min = self.view.spin_y_min.value()
            y_max = self.view.spin_y_max.value()
        
        self.view.display_seismic(self.current_data.T, x_range=(x_min, x_max), y_range=(y_min, y_max))
        self.update_contrast(); self.draw_horizons()
    

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
            im = ax.imshow(self.current_data, cmap=self.view.combo_cmap.currentText(), aspect='auto', extent=extent, vmin=-clip_val, vmax=clip_val, interpolation='bilinear')
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
            #Get current view range from the plot (what the user is looking at)
            view_range = self.view.plot_widget.viewRange()
            x_min, x_max = view_range[0]
            
            # Convert coordinates back to trace indices
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
            
            # Force reload at FULL RESOLUTION (step=1) for the attribute math
            # Not correct to calculate attributes on decimated data
            self.view.update_status(f"Fetching high-res data for {attr_type}...")
            QApplication.processEvents()
            
            # Update internal state so the 'Apply' button knows where the user is
            self.view.spin_x_min.setValue(x_min)
            self.view.spin_x_max.setValue(x_max)
            self.view.chk_manual_step.setChecked(True) 
            self.view.spin_step.setValue(1) # Force step 1
            
            # Perform the load
            self.load_data_internal(start_trace, end_trace, step=1, auto_fit=False)

        # Calculate the attribute on the fresh, high-res data
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
            
            # Refresh display
            self.update_display_only()
            self.view.update_status(f"Displayed: {attr_type} (High Res)")
            
        except Exception as e:
            QMessageBox.critical(self.view, "Attribute Error", str(e))
            self.view.update_status("Error calculating attribute")
    
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
            
            # Create figure 
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Calculate bin count (fewer bins = smoother histogram)
            # Ensure int
            n_bins_full = min(100, int(np.sqrt(len(amps))))  
            
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
            
            # Statistics box 
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
                    fontfamily='monospace', fontsize=11)  
            
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