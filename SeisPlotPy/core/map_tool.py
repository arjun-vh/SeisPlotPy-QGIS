from qgis.gui import QgsMapTool
from qgis.core import QgsPointXY
from qgis.PyQt.QtCore import pyqtSignal, Qt

class SeismicNavigationTool(QgsMapTool):
    """
    A simple tool that emits coordinates when the mouse moves or is double-clicked.
    """
    # Signal carrying the map coordinates (X, Y)
    canvas_moved = pyqtSignal(QgsPointXY)
    canvas_double_clicked = pyqtSignal(QgsPointXY)
    
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        # Set a crosshair cursor so the user knows the tool is active
        self.setCursor(Qt.CrossCursor)
        
    def canvasMoveEvent(self, event):
        """Triggered when mouse moves over the canvas"""
        # Convert screen pixels to map coordinates
        point = self.toMapCoordinates(event.pos())
        self.canvas_moved.emit(point)
        
    def canvasDoubleClickEvent(self, event):
        """Triggered on double click"""
        point = self.toMapCoordinates(event.pos())
        self.canvas_double_clicked.emit(point)