import sys
import os
import traceback
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QVBoxLayout, 
                             QDialog, QFormLayout, QLineEdit, QCheckBox, QPushButton, 
                             QHBoxLayout, QScrollArea, QWidget, QInputDialog, QGroupBox)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6 import uic

# Matplotlib PyQt6 Integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import your core engine components
from fea_engine import SimulationConfig, HardwareLibrary, run_simulation_job

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


class SimulationWorker(QThread):
    """
    Background thread to run the FEA simulation without freezing the GUI.
    """
    progress_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config, library, ref_name, emi_name, gask_name, finish):
        super().__init__()
        self.config = config
        self.library = library
        self.ref_name = ref_name
        self.emi_name = emi_name
        self.gask_name = gask_name
        self.finish = finish
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            fig, metrics = run_simulation_job(
                self.config, 
                self.library,
                self.ref_name,
                self.emi_name,
                self.gask_name,
                self.finish,
                log_callback=self.log_signal.emit, 
                progress_callback=self.progress_signal.emit,
                is_cancelled_callback=lambda: self._is_cancelled
            )
            
            if self._is_cancelled:
                self.log_signal.emit("\n[!] Simulation stopped by user.")
                self.finished_signal.emit(None, {})
            else:
                self.finished_signal.emit(fig, metrics)
                
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())


class SettingsDialog(QDialog):
    """
    Dynamic dialog that parses the active SimulationConfig object and generates
    text boxes and checkboxes, visually grouped by category.
    """
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.resize(550, 750)
        self.config = config
        
        # Categorized Mapping (Variable Name -> Human Readable Label)
        self.categories = {
            "Output & Rendering": {
                "generate_all_plots": "Generate All Plots (Batch Mode)",
                "plot_wall_shot": "Plot Wall Shot (2D Image)",
                "plot_intensity_x": "Plot Intensity Profile (X-Axis)",
                "plot_intensity_y": "Plot Intensity Profile (Y-Axis)",
                "plot_intensity_45": "Plot Intensity Profile (45° Diagonal)",
                "show_human_silhouette": "Show Human Silhouette Reference",
                "export_csv": "Export Results to CSV",
                "export_plots": "Export Plot Images",
                "batch_output_directory": "Output Directory Path"
            },
            "Simulation Space & Constraints": {
                "use_gpu": "Use GPU Acceleration (CUDA)",
                "max_multiple_reflections": "Max Multiple Reflections (Bounces)",
                "use_reflector_opening": "Force Reflector Opening Size",
                "target_distance_m": "Target Distance (meters)",
                "canvas_fov_deg": "Canvas Field of View (degrees)",
                "plot_fov_deg": "Plot Field of View (degrees)"
            },
            "Camera Settings": {
                "use_auto_exposure": "Use Auto Exposure",
                "auto_exposure_compensation_ev": "Auto Exposure Compensation (EV)",
                "cam_iso": "Camera ISO",
                "cam_f_stop": "Camera f-stop",
                "cam_shutter_speed_s": "Camera Shutter Speed (seconds)"
            },
            "Resolution & Angular Density": {
                "sim_grid_res": "Simulation Grid Resolution (px)",
                "sim_emitter_elements": "Emitter Subdivision Elements",
                "sim_theta_step_deg": "Theta Step Size (degrees)",
                "sim_phi_step_deg": "Phi Step Size (degrees)",
                "sim_theta_min_deg": "Theta Minimum (degrees)",
                "sim_theta_max_deg": "Theta Maximum (degrees)",
                "sim_phi_min_deg": "Phi Minimum (degrees)",
                "sim_phi_max_deg": "Phi Maximum (degrees)",
                "lumen_calc_step_deg": "Lumen Calculation Step (degrees)"
            },
            "Material Defaults & Thresholds": {
                "default_reflectivity_smooth": "Default Reflectivity (Smooth)",
                "default_reflectivity_op": "Default Reflectivity (Orange Peel)",
                "default_reflectivity_cylinder": "Default Reflectivity (Cylinder)",
                "default_op_blur_strength": "Orange Peel Blur Strength",
                "spill_visible_threshold_lux": "Spill Visible Threshold (Lux)",
                "corona_visible_threshold": "Corona Visible Threshold",
                "hotspot_fwhm_threshold": "Hotspot FWHM Threshold",
                "default_gasket_thickness_mm": "Default Gasket Thickness (mm)",
                "default_gasket_total_height_mm": "Default Gasket Total Height (mm)",
                "default_gasket_opening_mm": "Default Gasket Opening (mm)",
                "default_reflector_wall_thickness_mm": "Default Reflector Wall Thickness (mm)",
                "default_reflector_base_thickness_mm": "Default Reflector Base Thickness (mm)",
                "default_focus_offset_mm": "Default Focus Offset (mm)"
            }
        }

        self.main_layout = QVBoxLayout(self)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        
        self.input_widgets = {}
        self.populate_form()
        
        self.scroll.setWidget(self.scroll_widget)
        self.main_layout.addWidget(self.scroll)
        
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_save = QPushButton("Save Settings")
        
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_save)
        self.main_layout.addLayout(btn_layout)
        
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_reset.clicked.connect(self.reset_to_defaults)

    def populate_form(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.input_widgets.clear()
        
        for category_name, settings_map in self.categories.items():
            group_box = QGroupBox(category_name)
            font = group_box.font()
            font.setBold(True)
            group_box.setFont(font)
            
            group_layout = QFormLayout(group_box)
            
            for key, label_text in settings_map.items():
                value = getattr(self.config, key, None)
                if value is None and not isinstance(value, (bool, int, float, str)):
                    continue
                
                if isinstance(value, bool):
                    widget = QCheckBox()
                    widget.setChecked(value)
                else:
                    widget = QLineEdit(str(value))
                
                # Revert font to normal so text inputs aren't bold
                normal_font = widget.font()
                normal_font.setBold(False)
                widget.setFont(normal_font)
                
                group_layout.addRow(label_text, widget)
                self.input_widgets[key] = widget
                
            self.scroll_layout.addWidget(group_box)
            
        self.scroll_layout.addStretch()

    def save_settings(self):
        for key, widget in self.input_widgets.items():
            original_value = getattr(self.config, key)
            
            if isinstance(widget, QCheckBox):
                setattr(self.config, key, widget.isChecked())
            else:
                text_val = widget.text().strip()
                try:
                    if isinstance(original_value, int) and not isinstance(original_value, bool):
                        setattr(self.config, key, int(text_val))
                    elif isinstance(original_value, float):
                        setattr(self.config, key, float(text_val))
                    else:
                        setattr(self.config, key, text_val)
                except ValueError:
                    print(f"Warning: Could not parse '{text_val}' for setting '{key}'. Keeping previous value.")
                    
        self.config.save_settings()
        self.accept()

    def reset_to_defaults(self):
        reply = QMessageBox.question(self, "Confirm Reset", 
                                     "Are you sure you want to revert all settings to the default template?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(self.config.default_filepath):
                with open(self.config.default_filepath, 'r') as f:
                    data = json.load(f)
                
                # Unnest before setting attributes internally to prevent crashing
                for category, settings in data.items():
                    if isinstance(settings, dict):
                        for key, value in settings.items():
                            if key in ('active_emitter_name', 'active_reflector_name', 'active_gasket_name', 'reflector_finish'):
                                continue
                            setattr(self.config, key, value)
                    else:
                        setattr(self.config, category, settings)
                    
                self.populate_form()
            else:
                QMessageBox.warning(self, "Missing Template", f"Could not find '{self.config.default_filepath}'.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        ui_path = resource_path('mainwindow.ui')
        if not os.path.exists(ui_path):
            QMessageBox.critical(self, "Error", f"Could not find UI file: {ui_path}")
            sys.exit(1)
        uic.loadUi(ui_path, self)

        try:
            self.config = SimulationConfig(
                filepath=resource_path("simulation_settings.json"),
                default_filepath=resource_path("default_settings.json")
            )
            self.library = HardwareLibrary(
                filepath=resource_path("hardware_library.json")
            )
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        self.figure_canvas = None
        self.setup_canvas()
        self.setup_widget_mappings()
        self.populate_dropdowns()
        self.connect_signals()
        
        self.on_reflector_changed()
        self.on_emitter_changed()
        self.on_gasket_changed()

    def setup_canvas(self):
        self.lblPlotPlaceholder.hide()
        if self.grpPlot.layout() is None:
            self.grpPlot.setLayout(QVBoxLayout())

    def setup_widget_mappings(self):
        self.reflector_map = {
            "diameter_mm": self.txtRef_diameter_mm,
            "height_mm": self.txtRef_height_mm,
            "opening_diameter_mm": self.txtRef_opening_diameter_mm,
            "focus_offset_mm": self.txtRef_focus_offset_mm,
            "thickness_height_mm": self.txtRef_thickness_height_mm,
            "reflectivity_smooth": self.txtRef_reflectivity_smooth,
            "reflectivity_op": self.txtRef_reflectivity_op,
            "reflectivity_cylinder": self.txtRef_reflectivity_cylinder,
            "gasket_reflectivity": self.txtRef_gasket_reflectivity,
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
        self.cmbReflector.addItems(self.library.list_reflectors())
        self.cmbEmitter.addItems(self.library.list_emitters())
        self.cmbGasket.addItems(self.library.list_gaskets())

    def connect_signals(self):
        self.cmbReflector.currentIndexChanged.connect(self.on_reflector_changed)
        self.cmbEmitter.currentIndexChanged.connect(self.on_emitter_changed)
        self.cmbGasket.currentIndexChanged.connect(self.on_gasket_changed)

        # Reset Buttons
        self.btnResetReflector.clicked.connect(self.on_reflector_changed)
        self.btnResetEmitter.clicked.connect(self.on_emitter_changed)
        self.btnResetGasket.clicked.connect(self.on_gasket_changed)

        # Save and Delete Buttons
        self.btnSaveReflector.clicked.connect(self.save_reflector)
        self.btnDeleteReflector.clicked.connect(self.delete_reflector)
        self.btnSaveEmitter.clicked.connect(self.save_emitter)
        self.btnDeleteEmitter.clicked.connect(self.delete_emitter)
        self.btnSaveGasket.clicked.connect(self.save_gasket)
        self.btnDeleteGasket.clicked.connect(self.delete_gasket)
        
        # Bottom execution controls
        self.btnSettings.clicked.connect(self.open_settings)
        self.btnSimulate.clicked.connect(self.run_simulation)
        self.btnStop.clicked.connect(self.stop_simulation)

    # --- SAVE AND DELETE ABSTRACTION LOGIC ---

    def save_hardware_item(self, hw_type, current_name, widget_map, save_method, list_method, combo_box):
        new_name, ok = QInputDialog.getText(
            self, f"Save {hw_type}", f"Enter name for the {hw_type}:",
            QLineEdit.EchoMode.Normal, current_name
        )
        
        if ok and new_name.strip():
            new_name = new_name.strip()
            
            if new_name in list_method():
                reply = QMessageBox.question(
                    self, "Overwrite Confirm", 
                    f"A {hw_type} named '{new_name}' already exists. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
                    
            data = self.extract_fields_to_dict(widget_map)
            save_method(new_name, data)
            
            combo_box.blockSignals(True)
            combo_box.clear()
            combo_box.addItems(list_method())
            combo_box.setCurrentText(new_name)
            combo_box.blockSignals(False)
            
            self.log_message(f"Successfully saved {hw_type}: {new_name}")

    def delete_hardware_item(self, hw_type, current_name, delete_method, list_method, combo_box, reset_method):
        if not current_name:
            return
            
        reply = QMessageBox.question(
            self, "Delete Confirm", 
            f"Are you sure you want to permanently delete the {hw_type} '{current_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            delete_method(current_name)
            
            combo_box.blockSignals(True)
            combo_box.clear()
            combo_box.addItems(list_method())
            combo_box.blockSignals(False)
            
            reset_method() 
            self.log_message(f"Deleted {hw_type}: {current_name}")

    # --- SAVE / DELETE BUTTON HANDLERS ---

    def save_reflector(self):
        self.save_hardware_item("Reflector", self.cmbReflector.currentText(), self.reflector_map, 
                                self.library.add_or_update_reflector, self.library.list_reflectors, self.cmbReflector)
    def delete_reflector(self):
        self.delete_hardware_item("Reflector", self.cmbReflector.currentText(), 
                                  self.library.remove_reflector, self.library.list_reflectors, self.cmbReflector, self.on_reflector_changed)

    def save_emitter(self):
        self.save_hardware_item("Emitter", self.cmbEmitter.currentText(), self.emitter_map, 
                                self.library.add_or_update_emitter, self.library.list_emitters, self.cmbEmitter)
    def delete_emitter(self):
        self.delete_hardware_item("Emitter", self.cmbEmitter.currentText(), 
                                  self.library.remove_emitter, self.library.list_emitters, self.cmbEmitter, self.on_emitter_changed)

    def save_gasket(self):
        self.save_hardware_item("Gasket", self.cmbGasket.currentText(), self.gasket_map, 
                                self.library.add_or_update_gasket, self.library.list_gaskets, self.cmbGasket)
    def delete_gasket(self):
        self.delete_hardware_item("Gasket", self.cmbGasket.currentText(), 
                                  self.library.remove_gasket, self.library.list_gaskets, self.cmbGasket, self.on_gasket_changed)

    # --- ACTION LOGIC ---

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        dialog.exec()

    def populate_fields_from_dict(self, data_dict, widget_map):
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

    def extract_fields_to_dict(self, widget_map):
        extracted = {}
        for key, widget in widget_map.items():
            val_str = widget.text().strip()
            if not val_str:
                continue
                
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
        self.txtLogs.appendPlainText(message)
        scrollbar = self.txtLogs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, percent):
        self.progressBar.setValue(int(percent))

    def stop_simulation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.log_message("Sending interrupt signal to engine...")
            self.worker.cancel()
            self.btnStop.setEnabled(False)

    def run_simulation(self):
        ref_name = self.cmbReflector.currentText()
        emi_name = self.cmbEmitter.currentText()
        gask_name = self.cmbGasket.currentText()
        
        raw_finish = self.cmbReflectorFinish.currentText()
        finish_str = "smooth" if raw_finish == "Smooth" else "orange_peel"

        if not ref_name or not emi_name or not gask_name:
            QMessageBox.warning(self, "Missing Hardware", "Please select a Reflector, Emitter, and Gasket.")
            return

        overridden_ref = self.extract_fields_to_dict(self.reflector_map)
        overridden_emi = self.extract_fields_to_dict(self.emitter_map)
        overridden_gask = self.extract_fields_to_dict(self.gasket_map)

        self.library._reflectors[ref_name].update(overridden_ref)
        self.library._emitters[emi_name].update(overridden_emi)
        self.library._gaskets[gask_name].update(overridden_gask)
        
        self.config.generate_all_plots = False 

        self.btnSimulate.setEnabled(False)
        self.btnSettings.setEnabled(False)
        self.btnStop.setEnabled(True)
        
        self.progressBar.setValue(0)
        self.txtLogs.clear()
        self.log_message("--- INITIALIZING SIMULATION ---")

        self.worker = SimulationWorker(self.config, self.library, ref_name, emi_name, gask_name, finish_str)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.error_signal.connect(self.handle_simulation_error)
        self.worker.finished_signal.connect(self.handle_simulation_finished)
        self.worker.start()

    def handle_simulation_error(self, error_traceback):
        self.btnSimulate.setEnabled(True)
        self.btnSettings.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.log_message("CRITICAL ERROR IN ENGINE:")
        self.log_message(error_traceback)
        QMessageBox.critical(self, "Simulation Error", "An error occurred during simulation. Check logs.")

    def handle_simulation_finished(self, figure, metrics):
        self.btnSimulate.setEnabled(True)
        self.btnSettings.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.progressBar.setValue(100)
        
        if metrics:
            self.log_message("\n--- SIMULATION RESULTS ---")
            for k, v in metrics.items():
                self.log_message(f"{k}: {v}")

        if figure:
            if self.figure_canvas is not None:
                self.grpPlot.layout().removeWidget(self.figure_canvas)
                self.figure_canvas.deleteLater()
            
            self.figure_canvas = FigureCanvas(figure)
            self.grpPlot.layout().addWidget(self.figure_canvas)
            self.figure_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())