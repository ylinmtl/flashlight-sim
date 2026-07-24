import sys
import os
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QVBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6 import uic

# Matplotlib PyQt6 Integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import your core engine components
# Ensure 'fea_engine.py' is in the same directory.
from fea_engine import SimulationConfig, HardwareLibrary, run_simulation_job

class SimulationWorker(QThread):
    """
    Background thread to run the FEA simulation without freezing the GUI.
    Emits signals to update the UI safely across thread boundaries.
    """
    progress_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config, library):
        super().__init__()
        self.config = config
        self.library = library

    def run(self):
        try:
            # Execute the engine API. Pass our pyqtSignal emitters as the callbacks.
            fig, metrics = run_simulation_job(
                self.config, 
                self.library, 
                log_callback=self.log_signal.emit, 
                progress_callback=self.progress_signal.emit
            )
            self.finished_signal.emit(fig, metrics)
        except Exception as e:
            # Catch any math or CUDA errors and pass the stack trace to the GUI
            self.error_signal.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the UI file you generated
        ui_path = os.path.join(os.path.dirname(__file__), 'mainwindow.ui')
        if not os.path.exists(ui_path):
            QMessageBox.critical(self, "Error", f"Could not find UI file: {ui_path}")
            sys.exit(1)
        uic.loadUi(ui_path, self)

        # Initialize Data Engine
        try:
            self.config = SimulationConfig()
            self.library = HardwareLibrary()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        # Setup Matplotlib Canvas
        self.figure_canvas = None
        self.setup_canvas()

        # Map UI TextBoxes to JSON dictionary keys for easy data dumping/loading
        self.setup_widget_mappings()

        # Populate GUI state
        self.populate_dropdowns()
        self.connect_signals()
        
        # Force initial population of text boxes based on the first item in the dropdowns
        self.on_reflector_changed()
        self.on_emitter_changed()
        self.on_gasket_changed()

    def setup_canvas(self):
        """Replaces the static QLabel placeholder with a dynamic Matplotlib Canvas."""
        # Hide the placeholder label
        self.lblPlotPlaceholder.hide()
        
        # Create a layout for the plot area if it doesn't already have one
        if self.grpPlot.layout() is None:
            self.grpPlot.setLayout(QVBoxLayout())
            
        # We will instantiate the canvas later when the simulation finishes. 
        # For now, we just prepare the layout.

    def setup_widget_mappings(self):
        """Maps specific UI QLineEdits to the exact string keys expected by the FEA Engine."""
        self.reflector_map = {
            "diameter_mm": self.txtRef_diameter_mm,
            "height_mm": self.txtRef_height_mm,
            "opening_diameter_mm": self.txtRef_opening_diameter_mm,
            "focus_offset_mm": self.txtRef_focus_offset_mm,
            "thickness_height_mm": self.txtRef_thickness_height_mm,
            "reflectivity_smooth": self.txtRef_reflectivity_smooth,
            "reflectivity_op": self.txtRef_reflectivity_op,
            "reflectivity_cylinder": self.txtRef_reflectivity_cylinder,
            "OP_Factor": self.txtRef_OP_Factor
        }
        
        self.emitter_map = {
            "max_current_amps": self.txtEmi_max_current_amps,
            "vf_turn_on_v": self.txtEmi_vf_turn_on_v,
            "vf_scale": self.txtEmi_vf_scale,
            "base_efficacy_lm_w": self.txtEmi_base_efficacy_lm_w,
            "droop_factor": self.txtEmi_droop_factor,
            "footprint_x_mm": self.txtEmi_footprint_x_mm,
            "footprint_y_mm": self.txtEmi_footprint_y_mm,
            "height_mm": self.txtEmi_height_mm,
            "dome_size_mm": self.txtEmi_dome_size_mm,
            "refractive_index": self.txtEmi_refractive_index,
            "die_length_mm": self.txtEmi_die_length_mm,
            "die_width_mm": self.txtEmi_die_width_mm,
            "shape": self.txtEmi_shape
        }
        
        self.gasket_map = {
            "gasket_thickness_mm": self.txtGask_gasket_thickness_mm,
            "gasket_total_height_mm": self.txtGask_gasket_total_height_mm,
            "gasket_opening_mm": self.txtGask_gasket_opening_mm
        }

    def populate_dropdowns(self):
        """Loads available hardware names from the library into the UI Comboboxes."""
        self.cmbReflector.addItems(self.library.list_reflectors())
        self.cmbEmitter.addItems(self.library.list_emitters())
        self.cmbGasket.addItems(self.library.list_gaskets())

    def connect_signals(self):
        """Wires up the dropdown changes and button clicks to their logic functions."""
        # Dropdowns
        self.cmbReflector.currentIndexChanged.connect(self.on_reflector_changed)
        self.cmbEmitter.currentIndexChanged.connect(self.on_emitter_changed)
        self.cmbGasket.currentIndexChanged.connect(self.on_gasket_changed)

        # Reset Buttons
        self.btnResetReflector.clicked.connect(self.on_reflector_changed)
        self.btnResetEmitter.clicked.connect(self.on_emitter_changed)
        self.btnResetGasket.clicked.connect(self.on_gasket_changed)

        # Execution
        self.btnSimulate.clicked.connect(self.run_simulation)

    # --- AUTO-POPULATION LOGIC ---

    def populate_fields_from_dict(self, data_dict, widget_map):
        """Helper to safely push dictionary values into QLineEdits."""
        for key, widget in widget_map.items():
            val = data_dict.get(key, "")
            widget.setText(str(val))

    def on_reflector_changed(self):
        name = self.cmbReflector.currentText()
        if name:
            data = self.library.get_reflector(name)
            self.populate_fields_from_dict(data, self.reflector_map)

    def on_emitter_changed(self):
        name = self.cmbEmitter.currentText()
        if name:
            data = self.library.get_emitter(name)
            self.populate_fields_from_dict(data, self.emitter_map)

    def on_gasket_changed(self):
        name = self.cmbGasket.currentText()
        if name:
            data = self.library.get_gasket(name)
            self.populate_fields_from_dict(data, self.gasket_map)

    # --- EXECUTION LOGIC ---

    def extract_fields_to_dict(self, widget_map):
        """Helper to pull values from QLineEdits back into a dictionary with correct types."""
        extracted = {}
        for key, widget in widget_map.items():
            val_str = widget.text().strip()
            if not val_str:
                continue
                
            # Emitter shape is the only explicit string. Everything else parses to float.
            if key == "shape":
                extracted[key] = val_str
            else:
                try:
                    extracted[key] = float(val_str)
                except ValueError:
                    self.log_message(f"Warning: Could not parse '{val_str}' for {key}. Defaulting to 0.0")
                    extracted[key] = 0.0
        return extracted

    def log_message(self, message):
        """Appends a timestamped message to the GUI text box."""
        self.txtLogs.appendPlainText(message)
        # Scroll to bottom
        scrollbar = self.txtLogs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, percent):
        """Updates the progress bar safely from the worker thread."""
        self.progressBar.setValue(int(percent))

    def run_simulation(self):
        """Parses UI overrides, locks the GUI, and spawns the background worker thread."""
        ref_name = self.cmbReflector.currentText()
        emi_name = self.cmbEmitter.currentText()
        gask_name = self.cmbGasket.currentText()

        if not ref_name or not emi_name or not gask_name:
            QMessageBox.warning(self, "Missing Hardware", "Please select a Reflector, Emitter, and Gasket.")
            return

        # 1. Pull user overrides from the TextBoxes
        overridden_ref = self.extract_fields_to_dict(self.reflector_map)
        overridden_emi = self.extract_fields_to_dict(self.emitter_map)
        overridden_gask = self.extract_fields_to_dict(self.gasket_map)

        # 2. Inject overrides temporarily into the Hardware Library memory
        # (This avoids saving over the actual JSON unless explicitly desired)
        self.library._reflectors[ref_name].update(overridden_ref)
        self.library._emitters[emi_name].update(overridden_emi)
        self.library._gaskets[gask_name].update(overridden_gask)

        # 3. Inform the Config which profiles to use
        self.config.active_reflector_name = ref_name
        self.config.active_emitter_name = emi_name
        self.config.active_gasket_name = gask_name
        
        # Ensure single render mode for the GUI preview
        self.config.generate_all_plots = False 

        # 4. Lock the UI so the user doesn't spam the button
        self.btnSimulate.setEnabled(False)
        self.progressBar.setValue(0)
        self.txtLogs.clear()
        self.log_message("--- INITIALIZING SIMULATION ---")

        # 5. Spawn the Worker Thread
        self.worker = SimulationWorker(self.config, self.library)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.error_signal.connect(self.handle_simulation_error)
        self.worker.finished_signal.connect(self.handle_simulation_finished)
        self.worker.start()

    def handle_simulation_error(self, error_traceback):
        """Called if the FEA engine crashes or throws an exception."""
        self.btnSimulate.setEnabled(True)
        self.log_message("CRITICAL ERROR IN ENGINE:")
        self.log_message(error_traceback)
        QMessageBox.critical(self, "Simulation Error", "An error occurred during simulation. Check logs.")

    def handle_simulation_finished(self, figure, metrics):
        """Called when the worker thread successfully finishes processing."""
        self.btnSimulate.setEnabled(True)
        self.progressBar.setValue(100)
        
        # Format and log the returned metrics to the console
        self.log_message("\n--- SIMULATION RESULTS ---")
        for k, v in metrics.items():
            self.log_message(f"{k}: {v}")

        # Display the Matplotlib figure in the UI
        if figure:
            # If a canvas already exists from a previous run, clear it and remove it
            if self.figure_canvas is not None:
                self.grpPlot.layout().removeWidget(self.figure_canvas)
                self.figure_canvas.deleteLater()
            
            # Embed the newly returned figure
            self.figure_canvas = FigureCanvas(figure)
            self.grpPlot.layout().addWidget(self.figure_canvas)
            self.figure_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Optional: Set an application style (Fusion looks clean cross-platform)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())