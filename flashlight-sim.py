"""PyQt6 desktop front end for the flashlight beam simulator.

The window is laid out in mainwindow.ui and loaded at runtime. It offers a
catalogue browser for the three hardware kinds (reflector, emitter, gasket), a
settings dialog generated from the active SimulationConfig, and an embedded
Matplotlib canvas for the rendered beam. Simulations run on a worker thread so
the interface stays responsive and remains cancellable.
"""

import json
import os
import sys
import traceback

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QInputDialog, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QScrollArea,
                             QVBoxLayout, QWidget)

# Matplotlib's Qt canvas, used to embed the engine's figure in the window.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from fea_engine import (EmitterOffset, HardwareLibrary, SimulationConfig,
                        resource_path, run_simulation_job)

# Specs shown for each hardware kind, in the order they appear in the form. Each
# one maps to a QLineEdit in mainwindow.ui named <prefix><spec>, for example
# reflector "diameter_mm" -> txtRef_diameter_mm.
SPEC_FIELDS = {
    "reflector": ("diameter_mm", "height_mm", "opening_diameter_mm",
                  "focus_offset_mm", "thickness_diameter_mm", "thickness_height_mm",
                  "reflectivity_smooth", "reflectivity_op", "reflectivity_cylinder",
                  "gasket_reflectivity", "OP_Factor", "transmissivity_lens"),
    "emitter": ("max_current_amps", "vf_turn_on_v", "vf_scale", "base_efficacy_lm_w",
                "droop_factor", "footprint_x_mm", "footprint_y_mm", "height_mm",
                "dome_size_mm", "refractive_index", "die_length_mm", "die_width_mm",
                "shape", "die_outline"),
    "gasket": ("gasket_thickness_mm", "gasket_total_height_mm", "gasket_opening_mm"),
}

FIELD_WIDGET_PREFIX = {
    "reflector": "txtRef_",
    "emitter": "txtEmi_",
    "gasket": "txtGask_",
}

# Specs that stay strings; every other field is parsed as a float.
TEXT_SPECS = frozenset({"shape"})

# Specs held as a JSON array rather than a single value. The input box takes
# the text an outline generator produces, so a die shape can be pasted in.
LIST_SPECS = frozenset({"die_outline"})

# Reflector inputs that describe the build being simulated rather than the
# reflector itself, as (field, label). Each one is a QLineEdit in
# mainwindow.ui named <prefix><field>, exactly like a spec, but they are
# deliberately absent from SPEC_FIELDS so that saving a reflector discards
# them, and they reset to zero whenever the form is reloaded.
RUN_ONLY_REFLECTOR_FIELDS = (
    ("emitter_offset_distance_mm", "Emitter Offset Distance (mm)"),
    ("emitter_offset_angle_deg", "Emitter Offset Angle (° CW from up)"),
)

# Settings offered by the settings dialog, grouped exactly as they are stored,
# mapping each attribute of SimulationConfig to its human readable label.
SETTING_LABELS = {
    "Output & Rendering": {
        "generate_all_plots": "Generate All Plots (Batch Mode)",
        "plot_wall_shot": "Plot Wall Shot (2D Image)",
        "plot_intensity_x": "Plot Intensity Profile (X-Axis)",
        "plot_intensity_y": "Plot Intensity Profile (Y-Axis)",
        "plot_intensity_45": "Plot Intensity Profile (45° Diagonal)",
        "show_human_silhouette": "Show Human Silhouette Reference",
        "export_csv": "Export Results to CSV",
        "export_plots": "Export Plot Images",
        "batch_output_directory": "Output Directory Path",
    },
    "Simulation Space & Constraints": {
        "use_gpu": "Use GPU Acceleration (CUDA)",
        "max_multiple_reflections": "Max Multiple Reflections (Bounces)",
        "use_reflector_opening": "Force Reflector Opening Size",
        "target_distance_m": "Target Distance (meters)",
        "canvas_fov_deg": "Canvas Field of View (degrees)",
        "plot_fov_deg": "Plot Field of View (degrees)",
    },
    "Camera Settings": {
        "use_auto_exposure": "Use Auto Exposure",
        "auto_exposure_compensation_ev": "Auto Exposure Compensation (EV)",
        "cam_iso": "Camera ISO",
        "cam_f_stop": "Camera f-stop",
        "cam_shutter_speed_s": "Camera Shutter Speed (seconds)",
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
        "lumen_calc_step_deg": "Lumen Calculation Step (degrees)",
    },
    "Material Defaults & Thresholds": {
        "default_reflectivity_smooth": "Default Reflectivity (Smooth)",
        "default_reflectivity_op": "Default Reflectivity (Orange Peel)",
        "default_reflectivity_cylinder": "Default Reflectivity (Cylinder)",
        "default_gasket_reflectivity": "Default Reflectivity (Gasket)",
        "default_op_blur_strength": "Orange Peel Blur Strength",
        "default_op_factor": "Default OP Factor",
        "default_transmissivity_lens": "Default Lens Transmissivity",
        "spill_visible_threshold_lux": "Spill Visible Threshold (Lux)",
        "corona_visible_threshold": "Corona Visible Threshold",
        "hotspot_fwhm_threshold": "Hotspot FWHM Threshold",
        "default_gasket_thickness_mm": "Default Gasket Thickness (mm)",
        "default_gasket_total_height_mm": "Default Gasket Total Height (mm)",
        "default_gasket_opening_mm": "Default Gasket Opening (mm)",
        "default_reflector_wall_thickness_mm": "Default Reflector Wall Thickness (mm)",
        "default_reflector_base_thickness_mm": "Default Reflector Base Thickness (mm)",
        "default_focus_offset_mm": "Default Focus Offset (mm)",
        "default_opening_diameter_mm": "Default Reflector Opening Diameter (mm)",
        "default_dome_size_mm": "Default Emitter Dome Size (mm)",
        "default_refractive_index": "Default Emitter Refractive Index",
        "default_emitter_shape": "Default Emitter Die Shape",
    },
}


class SimulationWorker(QThread):
    """Runs one simulation job off the GUI thread.

    Signals:
        progress_signal: Completion percentage, 0-100.
        log_signal: A line of engine output.
        finished_signal: (figure, results) once the job ends; (None, {}) if it
            was cancelled.
        error_signal: Formatted traceback if the engine raised.
    """

    progress_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config, library, reflector_name, emitter_name, gasket_name,
                 finish, emitter_offset):
        """Captures everything the job needs; nothing is read from the GUI later.

        Args:
            config: Active SimulationConfig.
            library: Detached HardwareLibrary copy for this run, already
                carrying any unsaved edits from the form. It is never written,
                so nothing the worker does can reach hardware_library.json.
            reflector_name: Reflector to simulate.
            emitter_name: Emitter to simulate.
            gasket_name: Gasket to simulate.
            finish: "smooth" or "orange_peel".
            emitter_offset: EmitterOffset for this run. It is captured here
                rather than stored anywhere, so it lasts exactly one job.
        """
        super().__init__()
        self.config = config
        self.library = library
        self.reflector_name = reflector_name
        self.emitter_name = emitter_name
        self.gasket_name = gasket_name
        self.finish = finish
        self.emitter_offset = emitter_offset
        self._is_cancelled = False

    def cancel(self):
        """Asks the engine to stop at its next chunk boundary."""
        self._is_cancelled = True

    def run(self):
        """Runs the job and reports the outcome through the signals."""
        try:
            figure, results = run_simulation_job(
                self.config, self.library,
                self.reflector_name, self.emitter_name, self.gasket_name, self.finish,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit,
                is_cancelled_callback=lambda: self._is_cancelled,
                emitter_offset=self.emitter_offset)

            if self._is_cancelled:
                self.log_signal.emit("\n[!] Simulation stopped by user.")
                self.finished_signal.emit(None, {})
            else:
                self.finished_signal.emit(figure, results)

        except Exception:
            self.error_signal.emit(traceback.format_exc())


class SettingsDialog(QDialog):
    """Editor for every simulation setting, generated from SETTING_LABELS.

    Booleans become checkboxes and everything else a text box. Edits are written
    back to the config with the type the setting already had, so a value that
    was loaded as an int stays an int.
    """

    def __init__(self, config, parent=None):
        """Builds the scrollable form for the given config.

        Args:
            config: SimulationConfig to edit in place.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.resize(550, 750)
        self.config = config
        self.input_widgets = {}

        self.scroll_layout = QVBoxLayout()
        scroll_contents = QWidget()
        scroll_contents.setLayout(self.scroll_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_contents)

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_to_defaults)
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)

        button_row = QHBoxLayout()
        button_row.addWidget(reset_button)
        button_row.addWidget(save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)
        layout.addLayout(button_row)

        self.populate_form()

    def populate_form(self):
        """Rebuilds every input from the current config values."""
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.input_widgets.clear()

        for category, labels in SETTING_LABELS.items():
            group_box = QGroupBox(category)
            bold_font = group_box.font()
            bold_font.setBold(True)
            group_box.setFont(bold_font)
            form = QFormLayout(group_box)

            for key, label in labels.items():
                value = getattr(self.config, key, None)
                if value is None:
                    continue  # Setting is absent from this config file.

                if isinstance(value, bool):
                    widget = QCheckBox()
                    widget.setChecked(value)
                else:
                    widget = QLineEdit(str(value))

                # Undo the group box's bold font, which children inherit.
                normal_font = widget.font()
                normal_font.setBold(False)
                widget.setFont(normal_font)

                form.addRow(label, widget)
                self.input_widgets[key] = widget

            self.scroll_layout.addWidget(group_box)

        self.scroll_layout.addStretch()

    def save_settings(self):
        """Writes every input back to the config, then to disk, and closes."""
        for key, widget in self.input_widgets.items():
            previous_value = getattr(self.config, key)

            if isinstance(widget, QCheckBox):
                setattr(self.config, key, widget.isChecked())
                continue

            text = widget.text().strip()
            try:
                # bools are excluded because in Python they are also ints.
                if isinstance(previous_value, int) and not isinstance(previous_value, bool):
                    setattr(self.config, key, int(text))
                elif isinstance(previous_value, float):
                    setattr(self.config, key, float(text))
                else:
                    setattr(self.config, key, text)
            except ValueError:
                print(f"Warning: Could not parse '{text}' for setting '{key}'. "
                      f"Keeping previous value.")

        self.config.save_settings()
        self.accept()

    def reset_to_defaults(self):
        """Reloads every tunable from the shipped template, after confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Are you sure you want to revert all settings to the default template?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.config.reset_to_defaults()
        except FileNotFoundError as error:
            QMessageBox.warning(self, "Missing Template", str(error))
            return

        self.populate_form()


class MainWindow(QMainWindow):
    """Main window: hardware catalogues, simulation controls and results."""

    def __init__(self):
        """Loads the UI, the config and the hardware library, then wires it up."""
        super().__init__()

        ui_path = resource_path("mainwindow.ui")
        if not os.path.exists(ui_path):
            QMessageBox.critical(self, "Error", f"Could not find UI file: {ui_path}")
            sys.exit(1)
        uic.loadUi(ui_path, self)

        try:
            self.config = SimulationConfig()
            self.library = HardwareLibrary()
            # A newer release may have added whole hardware entries, or added
            # specs to entries the operator already has. Both are compared
            # against the shipped copies and applied silently.
            self.imported_entries = self.library.import_new_entries()
            self.restored_specs = self.library.restore_missing_specs(self.config)
        except Exception as error:
            QMessageBox.critical(self, "Initialization Error", str(error))
            sys.exit(1)

        self.figure_canvas = None
        self.worker = None

        self.setup_canvas()
        self.setup_hardware_widgets()
        self.connect_signals()

        for kind in SPEC_FIELDS:
            self.reload_fields(kind)

        # Both files are upgraded silently on load; say so, because the
        # operator is about to simulate with values they never chose.
        if self.config.restored_settings:
            self.log_message(
                f"Settings file upgraded: {len(self.config.restored_settings)} new "
                f"setting(s) taken from the template "
                f"({', '.join(self.config.restored_settings)}).")

        if self.imported_entries:
            added_count = sum(len(names) for names in self.imported_entries.values())
            self.log_message(
                f"Hardware library upgraded: {added_count} new entrie(s) added "
                f"from the shipped library.")
            for kind, names in sorted(self.imported_entries.items()):
                self.log_message(f"  {kind}: {', '.join(names)}")

        if self.restored_specs:
            restored_count = sum(len(specs) for specs in self.restored_specs.values())
            self.log_message(
                f"Hardware library upgraded: {restored_count} missing spec(s) "
                f"filled in from the settings across "
                f"{len(self.restored_specs)} entrie(s).")
            for entry, specs in sorted(self.restored_specs.items()):
                self.log_message(f"  {entry}: {', '.join(specs)}")

    # --- SETUP ---

    def setup_canvas(self):
        """Prepares the plot area to receive a Matplotlib canvas."""
        self.lblPlotPlaceholder.hide()
        if self.grpPlot.layout() is None:
            self.grpPlot.setLayout(QVBoxLayout())

    def setup_hardware_widgets(self):
        """Indexes the combo boxes and spec inputs by hardware kind."""
        self.combo_boxes = {
            "reflector": self.cmbReflector,
            "emitter": self.cmbEmitter,
            "gasket": self.cmbGasket,
        }
        self.field_widgets = {
            kind: {field: getattr(self, FIELD_WIDGET_PREFIX[kind] + field)
                   for field in fields}
            for kind, fields in SPEC_FIELDS.items()
        }
        # Kept apart from field_widgets so that read_fields, and therefore
        # saving, never sees them.
        self.run_only_widgets = {
            "reflector": {
                field: getattr(self, FIELD_WIDGET_PREFIX["reflector"] + field)
                for field, _ in RUN_ONLY_REFLECTOR_FIELDS
            },
        }
        for kind, combo in self.combo_boxes.items():
            combo.addItems(self.library.names(kind))

    def connect_signals(self):
        """Connects the catalogue buttons and the bottom control bar."""
        for kind, save_button, delete_button, reset_button in (
                ("reflector", self.btnSaveReflector, self.btnDeleteReflector,
                 self.btnResetReflector),
                ("emitter", self.btnSaveEmitter, self.btnDeleteEmitter,
                 self.btnResetEmitter),
                ("gasket", self.btnSaveGasket, self.btnDeleteGasket,
                 self.btnResetGasket)):
            # k=kind binds the current value; the signals' own arguments are
            # swallowed by *_ because none of these slots need them.
            self.combo_boxes[kind].currentIndexChanged.connect(
                lambda *_, k=kind: self.reload_fields(k))
            reset_button.clicked.connect(lambda *_, k=kind: self.reload_fields(k))
            save_button.clicked.connect(lambda *_, k=kind: self.save_hardware(k))
            delete_button.clicked.connect(lambda *_, k=kind: self.delete_hardware(k))

        self.btnSettings.clicked.connect(self.open_settings)
        self.btnSimulate.clicked.connect(self.run_simulation)
        self.btnStop.clicked.connect(self.stop_simulation)

    # --- HARDWARE CATALOGUE ---

    def reload_fields(self, kind):
        """Fills the spec inputs from the catalogue entry now selected.

        Every optional spec is filled in at start up, so nothing shows blank
        unless the entry is genuinely missing a mandatory spec. The run-only
        inputs go back to zero, since the catalogue holds no value for them to
        be restored from.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        for widget in self.run_only_widgets.get(kind, {}).values():
            widget.setText("0.0")

        name = self.combo_boxes[kind].currentText()
        if not name:
            return

        specs = self.library.get(kind, name)
        for field, widget in self.field_widgets[kind].items():
            value = specs.get(field, "")
            if field in LIST_SPECS and value != "":
                # Compact JSON, so the box holds exactly what a generator emits
                # and can be copied back out again.
                value = json.dumps(value, separators=(",", ":"))
            widget.setText(str(value))

    def read_fields(self, kind):
        """Reads the spec inputs back into a specs dict.

        Blank inputs are omitted so the stored value survives. Unparseable
        numbers fall back to 0.0 and are reported in the log.

        Args:
            kind: One of the keys of SPEC_FIELDS.

        Returns:
            The specs the operator currently has on screen.
        """
        specs = {}
        for field, widget in self.field_widgets[kind].items():
            text = widget.text().strip()
            if not text:
                continue

            if field in TEXT_SPECS:
                specs[field] = text
                continue

            if field in LIST_SPECS:
                try:
                    specs[field] = json.loads(text)
                except json.JSONDecodeError as error:
                    self.log_message(f"Warning: Could not parse {field} as JSON "
                                     f"({error}). Keeping the stored value.")
                continue

            try:
                specs[field] = float(text)
            except ValueError:
                self.log_message(f"Warning: Could not parse '{text}' for {field}. "
                                 f"Defaulting to 0.0")
                specs[field] = 0.0
        return specs

    def read_emitter_offset(self):
        """Reads the run-only emitter centring offset from the Reflector column.

        Returns:
            An EmitterOffset in polar form. A blank or unparseable box reads as
            zero, so a typo simulates a centred emitter rather than stopping
            the run, and is reported in the log.
        """
        values = {}
        for field, widget in self.run_only_widgets["reflector"].items():
            text = widget.text().strip()
            try:
                values[field] = float(text) if text else 0.0
            except ValueError:
                self.log_message(f"Warning: Could not parse '{text}' for {field}. "
                                 f"Defaulting to 0.0")
                values[field] = 0.0

        return EmitterOffset(values["emitter_offset_distance_mm"],
                             values["emitter_offset_angle_deg"])

    def refresh_dropdown(self, kind, select=None):
        """Reloads one dropdown from the library without firing its signals.

        Args:
            kind: One of the keys of SPEC_FIELDS.
            select: Entry to select afterwards, or None to leave the selection
                to Qt.
        """
        combo = self.combo_boxes[kind]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(self.library.names(kind))
        if select is not None:
            combo.setCurrentText(select)
        combo.blockSignals(False)

    def save_hardware(self, kind):
        """Saves the on-screen specs to the catalogue under a chosen name.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        label = kind.capitalize()
        new_name, confirmed = QInputDialog.getText(
            self, f"Save {label}", f"Enter name for the {label}:",
            QLineEdit.EchoMode.Normal, self.combo_boxes[kind].currentText())

        if not confirmed or not new_name.strip():
            return
        new_name = new_name.strip()

        if new_name in self.library.names(kind):
            reply = QMessageBox.question(
                self, "Overwrite Confirm",
                f"A {label} named '{new_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self.library.save(kind, new_name, self.read_fields(kind))
        self.refresh_dropdown(kind, select=new_name)
        self.log_message(f"Successfully saved {label}: {new_name}")

    def delete_hardware(self, kind):
        """Deletes the selected catalogue entry after confirmation.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        label = kind.capitalize()
        name = self.combo_boxes[kind].currentText()
        if not name:
            return

        reply = QMessageBox.question(
            self, "Delete Confirm",
            f"Are you sure you want to permanently delete the {label} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.library.delete(kind, name)
        self.refresh_dropdown(kind)
        self.reload_fields(kind)
        self.log_message(f"Deleted {label}: {name}")

    # --- SIMULATION ---

    def open_settings(self):
        """Opens the settings dialog modally."""
        SettingsDialog(self.config, self).exec()

    def log_message(self, message):
        """Appends a line to the log pane and scrolls to it."""
        self.txtLogs.appendPlainText(message)
        scrollbar = self.txtLogs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, percent):
        """Moves the progress bar."""
        self.progressBar.setValue(int(percent))

    def set_controls_running(self, is_running):
        """Enables the stop button and disables the rest while a job runs."""
        self.btnSimulate.setEnabled(not is_running)
        self.btnSettings.setEnabled(not is_running)
        self.btnStop.setEnabled(is_running)

    def run_simulation(self):
        """Validates the selection, applies edits and starts the worker."""
        names = {kind: combo.currentText() for kind, combo in self.combo_boxes.items()}
        if not all(names.values()):
            QMessageBox.warning(self, "Missing Hardware",
                                "Please select a Reflector, Emitter, and Gasket.")
            return

        finish = ("smooth" if self.cmbReflectorFinish.currentText() == "Smooth"
                  else "orange_peel")

        # The run works on a detached copy of the catalogue. On-screen edits are
        # overlaid onto that copy, so they apply to this simulation and to
        # nothing else: the live catalogue is untouched, and only the Save
        # button ever changes hardware_library.json.
        run_library = self.library.copy_for_run()
        for kind, name in names.items():
            run_library.apply_overrides(kind, name, self.read_fields(kind))

        # The centring offset never reaches the catalogue at all, not even the
        # run copy; it is passed straight to the job.
        emitter_offset = self.read_emitter_offset()

        # The Run button always renders the current selection. Batch mode is
        # driven from the settings dialog instead.
        self.config.generate_all_plots = False

        self.set_controls_running(True)
        self.progressBar.setValue(0)
        self.txtLogs.clear()
        self.log_message("--- INITIALIZING SIMULATION ---")

        self.worker = SimulationWorker(self.config, run_library, names["reflector"],
                                       names["emitter"], names["gasket"], finish,
                                       emitter_offset)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.error_signal.connect(self.handle_simulation_error)
        self.worker.finished_signal.connect(self.handle_simulation_finished)
        self.worker.start()

    def stop_simulation(self):
        """Asks a running job to stop at its next chunk boundary."""
        if self.worker is not None and self.worker.isRunning():
            self.log_message("Sending interrupt signal to engine...")
            self.worker.cancel()
            self.btnStop.setEnabled(False)

    def handle_simulation_error(self, error_traceback):
        """Restores the controls and reports an engine crash.

        Args:
            error_traceback: Formatted traceback from the worker.
        """
        self.set_controls_running(False)
        self.log_message("CRITICAL ERROR IN ENGINE:")
        self.log_message(error_traceback)
        QMessageBox.critical(self, "Simulation Error",
                             "An error occurred during simulation. Check logs.")

    def handle_simulation_finished(self, figure, results):
        """Restores the controls, logs the results and shows the new plot.

        Args:
            figure: Matplotlib figure to display, or None.
            results: Headline results keyed by label; empty if cancelled.
        """
        self.set_controls_running(False)
        self.progressBar.setValue(100)

        if results:
            self.log_message("\n--- SIMULATION RESULTS ---")
            for label, value in results.items():
                self.log_message(f"{label}: {value}")

        if figure:
            if self.figure_canvas is not None:
                self.grpPlot.layout().removeWidget(self.figure_canvas)
                self.figure_canvas.deleteLater()

            self.figure_canvas = FigureCanvas(figure)
            self.grpPlot.layout().addWidget(self.figure_canvas)
            self.figure_canvas.draw()


def main():
    """Starts the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()