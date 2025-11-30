import sys
import subprocess
import os
from qgis.PyQt.QtWidgets import QMessageBox

def get_python_interpreter():
    """
    Finds the real python executable.
    In QGIS on Windows, sys.executable points to qgis-bin.exe, which cannot run pip.
    """
    executable = sys.executable
    
    # If it looks like python, return it
    if os.path.basename(executable).lower().startswith('python'):
        return executable
        
    # If not (e.g. qgis.exe), find python in the prefix
    # Windows: python.exe is usually in sys.exec_prefix
    if os.name == 'nt':
        candidate = os.path.join(sys.exec_prefix, 'python.exe')
        if os.path.exists(candidate):
            return candidate
            
    # Linux/Mac: python3 is usually in sys.exec_prefix/bin
    candidate = os.path.join(sys.exec_prefix, 'bin', 'python3')
    if os.path.exists(candidate):
        return candidate
        
    # Fallback (this might fail if not in path)
    return 'python'

def install_dependencies(iface):
    """
    Checks for required packages and installs them using pip if missing.
    """
    required = {
        'segyio': 'segyio',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pyqtgraph': 'pyqtgraph',
        'numpy': 'numpy'
    }
    
    missing = []
    
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
            
    if not missing:
        return True

    msg = (
        "The 'SeisPlotPy' plugin requires the following missing Python libraries:\n\n"
        f"{', '.join(missing)}\n\n"
        "Do you want to download and install them now?\n"
        "(This will use the standard Python installer 'pip')"
    )
    
    reply = QMessageBox.question(
        iface.mainWindow(), 
        "Missing Dependencies", 
        msg, 
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.Yes
    )
    
    if reply == QMessageBox.No:
        return False

    python_exec = get_python_interpreter()
    
    # Command: python.exe -m pip install --user package_name
    cmd = [python_exec, '-m', 'pip', 'install', '--user'] + missing
    
    try:
        iface.messageBar().pushMessage("Installing libraries...", "Please wait...", level=0, duration=0)
        
        # This prevents the black command window from popping up on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        subprocess.check_call(cmd, startupinfo=startupinfo)
        
        QMessageBox.information(
            iface.mainWindow(), 
            "Success", 
            "Libraries installed successfully!\n\nPlease RESTART QGIS to activate the plugin."
        )
        return False
        
    except subprocess.CalledProcessError as e:
        QMessageBox.critical(
            iface.mainWindow(), 
            "Installation Failed", 
            f"Could not install libraries automatically.\n\nError Code: {e.returncode}\nCommand: {' '.join(cmd)}\n\nPlease install manually."
        )
        return False
    except Exception as e:
        QMessageBox.critical(iface.mainWindow(), "Error", f"Unexpected error: {str(e)}")
        return False