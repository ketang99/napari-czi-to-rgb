"""
plugin.py
---------
Napari dock widget for viewing CZI files.
Run this file directly to launch napari with the plugin loaded.

Usage:
    python plugin.py
"""

import napari
import numpy as np
import tifffile
import os
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel,
    QSlider, QFileDialog, QComboBox,
    QGroupBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup,
)
from qtpy.QtCore import Qt
 
import czi_processing as cp

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_scenes: dict = {}            # {int: np.ndarray (Y, X, 4)}  — original
_scenes_rgb: dict = {}        # {int: np.ndarray (Y, X, 3)}  — converted
_metadata: dict = {}
_channel_names: list = []
_current_scene: int = 0
_view_mode: str = "original"  # "original" or "rgb"
_last_conversion_params: dict = {}
 
COLORMAPS_ORIGINAL = ["blue", "green", "red", "magenta"]
COLORMAPS_RGB      = ["blue", "green", "red"]
NAMES_RGB          = ["Blue (B)", "Green (G)", "Red (R)"]


# ---------------------------------------------------------------------------
# Helper: display a scene in the napari viewer
# ---------------------------------------------------------------------------
def _display_scene(viewer: napari.Viewer, scene_idx: int) -> None:
    global _current_scene
    _current_scene = scene_idx
 
    # Save visibility state before clearing
    n_expected = 3 if _view_mode == "rgb" else 4
    if len(viewer.layers) == n_expected:
        visibility = [viewer.layers[c].visible for c in range(n_expected)]
    else:
        visibility = [True] * n_expected
 
    viewer.layers.clear()
 
    if _view_mode == "rgb" and _scenes_rgb:
        arr = _scenes_rgb[scene_idx]   # (Y, X, 3)
        for c, (cmap, name) in enumerate(zip(COLORMAPS_RGB, NAMES_RGB)):
            viewer.add_image(
                arr[:, :, c],
                name=name,
                colormap=cmap,
                blending="additive",
                visible=visibility[c],
            )
    else:
        arr = _scenes[scene_idx]       # (Y, X, 4)
        for c, cmap in enumerate(COLORMAPS_ORIGINAL):
            name = _channel_names[c] if c < len(_channel_names) else f"Ch {c}"
            viewer.add_image(
                arr[:, :, c],
                name=name,
                colormap=cmap,
                blending="additive",
                visible=visibility[c],
            )


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------
class CZIViewerWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)
        self.setLayout(root)

        # ── File path row ──────────────────────────────────────────────
        root.addWidget(QLabel("CZI file path:"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select or type a .czi path…")
        self.path_edit.textChanged.connect(self._on_path_changed)
        path_row.addWidget(self.path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        # ── Load button ────────────────────────────────────────────────
        self.load_btn = QPushButton("Load")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load)
        root.addWidget(self.load_btn)

        # ── Status label ───────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # ── Scene navigation ───────────────────────────────────────────
        self.nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(6)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_widget.setLayout(nav_layout)
 
        self.scene_label = QLabel("Scene 1 / ?")
        self.scene_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.scene_label)
 
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider)
        nav_layout.addWidget(self.slider)
 
        arrow_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀  Prev")
        self.next_btn = QPushButton("Next  ▶")
        self.prev_btn.clicked.connect(self._prev_scene)
        self.next_btn.clicked.connect(self._next_scene)
        arrow_row.addWidget(self.prev_btn)
        arrow_row.addWidget(self.next_btn)
        nav_layout.addLayout(arrow_row)
 
        self.nav_widget.setVisible(False)
        root.addWidget(self.nav_widget)
 
        # ── RGB Conversion group ───────────────────────────────────────
        self.rgb_group = QGroupBox("RGB Conversion")
        self.rgb_group.setVisible(False)
        rgb_layout = QVBoxLayout()
        rgb_layout.setSpacing(6)
        self.rgb_group.setLayout(rgb_layout)
 
        # convert_mode
        rgb_layout.addWidget(QLabel("Convert mode:"))
        self.combo_convert_mode = QComboBox()
        self.combo_convert_mode.addItems(["Remove647", "MergeRed", "MergeMagenta"])
        rgb_layout.addWidget(self.combo_convert_mode)
 
        # norm_mode
        rgb_layout.addWidget(QLabel("Norm mode:"))
        self.combo_norm_mode = QComboBox()
        self.combo_norm_mode.addItems(["DirectMinMax", "Percentile"])
        self.combo_norm_mode.currentTextChanged.connect(self._on_norm_mode_changed)
        rgb_layout.addWidget(self.combo_norm_mode)
 
        # percentile input (enabled only when Percentile is selected)
        pctile_row = QHBoxLayout()
        self.pctile_label = QLabel("Percentile:")
        self.pctile_spin = QDoubleSpinBox()
        self.pctile_spin.setRange(0.0, 49.9)
        self.pctile_spin.setSingleStep(0.5)
        self.pctile_spin.setValue(1.0)
        self.pctile_spin.setEnabled(False)
        self.pctile_label.setEnabled(False)
        pctile_row.addWidget(self.pctile_label)
        pctile_row.addWidget(self.pctile_spin)
        rgb_layout.addLayout(pctile_row)
 
        # norm_before_combine
        rgb_layout.addWidget(QLabel("Norm before combine:"))
        self.combo_norm_before = QComboBox()
        self.combo_norm_before.addItems(["False", "True"])
        rgb_layout.addWidget(self.combo_norm_before)
 
        # norm_after_combine
        rgb_layout.addWidget(QLabel("Norm after combine:"))
        self.combo_norm_after = QComboBox()
        self.combo_norm_after.addItems(["True", "False"])
        rgb_layout.addWidget(self.combo_norm_after)
 
        # Convert button
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self._convert)
        rgb_layout.addWidget(self.convert_btn)
 
        # Conversion status
        self.convert_status = QLabel("")
        self.convert_status.setWordWrap(True)
        rgb_layout.addWidget(self.convert_status)
 
        # View toggle — original vs rgb
        rgb_layout.addWidget(QLabel("Display:"))
        toggle_row = QHBoxLayout()
        self.radio_original = QRadioButton("Original")
        self.radio_rgb      = QRadioButton("RGB")
        self.radio_original.setChecked(True)
        self.radio_original.setEnabled(False)
        self.radio_rgb.setEnabled(False)
        self.radio_original.toggled.connect(self._on_view_toggle)
        self.view_toggle_group = QButtonGroup()
        self.view_toggle_group.addButton(self.radio_original)
        self.view_toggle_group.addButton(self.radio_rgb)
        toggle_row.addWidget(self.radio_original)
        toggle_row.addWidget(self.radio_rgb)
        rgb_layout.addLayout(toggle_row)
 
        # Save TIFF button
        self.save_btn = QPushButton("Save as TIFF")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_tiff)
        rgb_layout.addWidget(self.save_btn)
 
        root.addWidget(self.rgb_group)
        root.addStretch()

    # ------------------------------------------------------------------
    # Slots - file loading
    # ------------------------------------------------------------------
    def _on_path_changed(self, text: str):
        valid = text.strip().endswith(".czi") and len(text.strip()) > 4
        self.load_btn.setEnabled(valid)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CZI file", "", "CZI Files (*.czi)"
        )
        if path:
            self.path_edit.setText(path)

    def _load(self):
        global _scenes, _metadata, _channel_names, _scenes_rgb, _view_mode
 
        path = self.path_edit.text().strip()
        self.status_label.setText("Loading…")
        self.load_btn.setEnabled(False)
        self.repaint()
 
        try:
            _scenes, _metadata = cp.load_czi(path)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.load_btn.setEnabled(True)
            return
 
        # Extract channel names from metadata
        try:
            channels = _metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]
            _channel_names = [ch["@Name"] for ch in channels]
        except Exception:
            _channel_names = [f"Ch {c}" for c in range(4)]
 
        # Reset RGB state
        _scenes_rgb = {}
        _view_mode = "original"
        self.radio_original.setChecked(True)
        self.radio_original.setEnabled(False)
        self.radio_rgb.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.convert_status.setText("")
 
        n = len(_scenes)
        self.status_label.setText(f"Loaded {n} scene(s).")
 
        self.slider.setMaximum(n - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(max(1, n // 10))
        self._update_nav_label(0, n)
        self.nav_widget.setVisible(True)
        self.rgb_group.setVisible(True)
        self.load_btn.setEnabled(True)
 
        _display_scene(self.viewer, 0)
        self._update_arrow_states(0, n)
 
    # ------------------------------------------------------------------
    # Slots — scene navigation
    # ------------------------------------------------------------------
    def _on_slider(self, value: int):
        n = len(_scenes)
        if n == 0:
            return
        self._update_nav_label(value, n)
        _display_scene(self.viewer, value)
        self._update_arrow_states(value, n)
 
    def _prev_scene(self):
        v = self.slider.value()
        if v > 0:
            self.slider.setValue(v - 1)
 
    def _next_scene(self):
        v = self.slider.value()
        if v < self.slider.maximum():
            self.slider.setValue(v + 1)
 
    # ------------------------------------------------------------------
    # Slots — RGB conversion
    # ------------------------------------------------------------------
    def _on_norm_mode_changed(self, text: str):
        is_pctile = text == "Percentile"
        self.pctile_spin.setEnabled(is_pctile)
        self.pctile_label.setEnabled(is_pctile)
 
    def _convert(self):
        global _scenes_rgb, _view_mode, _last_conversion_params
 
        self.convert_status.setText("Converting…")
        self.convert_btn.setEnabled(False)
        self.repaint()
 
        params = {
            "convert_mode":        self.combo_convert_mode.currentText(),
            "norm_mode":           self.combo_norm_mode.currentText(),
            "pctile_value":        self.pctile_spin.value() if self.combo_norm_mode.currentText() == "Percentile" else None,
            "norm_before_combine": self.combo_norm_before.currentText() == "True",
            "norm_after_combine":  self.combo_norm_after.currentText() == "True",
        }
 
        try:
            _scenes_rgb = cp.convert_to_rgb_all_scenes(_scenes, params)
            _last_conversion_params = params
        except Exception as e:
            self.convert_status.setText(f"Error: {e}")
            self.convert_btn.setEnabled(True)
            return
 
        self.convert_status.setText("Conversion done.")
        self.convert_btn.setEnabled(True)
        self.radio_original.setEnabled(True)
        self.radio_rgb.setEnabled(True)
        self.save_btn.setEnabled(True)
 
        # Auto-switch view to RGB after conversion
        _view_mode = "rgb"
        self.radio_rgb.setChecked(True)
        _display_scene(self.viewer, _current_scene)
 
    def _on_view_toggle(self):
        global _view_mode
        _view_mode = "original" if self.radio_original.isChecked() else "rgb"
        _display_scene(self.viewer, _current_scene)
 
    # ------------------------------------------------------------------
    # Slots — save
    # ------------------------------------------------------------------
    def _save_tiff(self):
        if not _scenes_rgb:
            return
 
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save TIFF", "", "TIFF Files (*.tif *.tiff)"
        )
        if not save_path:
            return
 
        if not save_path.lower().endswith((".tif", ".tiff")):
            save_path += ".tif"
 
        try:
            # Write each scene as a page — scenes may differ in (Y, X) size
            # so we can't stack into a single array; write individually instead
            with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
                for k in sorted(_scenes_rgb.keys()):
                    tif.write(_scenes_rgb[k][...,-1::-1], contiguous=False)
 
            # Save conversion params as a .txt alongside the tiff
            params_path = os.path.splitext(save_path)[0] + "_conversion_params.txt"
            with open(params_path, "w") as f:
                f.write("CZI RGB Conversion Parameters\n")
                f.write("=" * 35 + "\n")
                for key, val in _last_conversion_params.items():
                    f.write(f"{key}: {val}\n")
                f.write(f"\nSource file: {self.path_edit.text().strip()}\n")
                f.write(f"Number of scenes: {len(_scenes_rgb)}\n")
 
            self.convert_status.setText(f"Saved to {os.path.basename(save_path)}")
 
        except Exception as e:
            self.convert_status.setText(f"Save error: {e}")
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_nav_label(self, idx: int, total: int):
        self.scene_label.setText(f"Scene {idx + 1} / {total}")
 
    def _update_arrow_states(self, idx: int, total: int):
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < total - 1)
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    viewer = napari.Viewer()
    widget = CZIViewerWidget(viewer)
    viewer.window.add_dock_widget(widget, name="CZI Viewer", area="right")
    napari.run()
 
 
if __name__ == "__main__":
    main()
 