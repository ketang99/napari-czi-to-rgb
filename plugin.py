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
import traceback
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel,
    QSlider, QFileDialog, QComboBox,
    QGroupBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QGridLayout,
)
from qtpy.QtCore import Qt
 
import czi_processing as cp

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_scenes: dict = {}            # {int: np.ndarray (C, Z, Y, X)} — original
_scenes_rgb: dict = {}        # {int: np.ndarray (C, Z, Y, X)} — converted BGR
_metadata: dict = {}
_channel_names: list = []
_scene_keys: list = []
_current_scene: int = 0
_current_z: int = 0
_view_mode: str = "original"  # "original" or "rgb"
_conversion_params: dict = {}
_last_conversion_params: dict = {}
_channel_axis: int = 0
 
COLORMAPS_ORIGINAL = ["blue", "green", "red", "magenta"]
COLORMAPS_RGB      = ["blue", "green", "red"]
NAMES_RGB          = ["Blue (B)", "Green (G)", "Red (R)"]
_visibility_original: list = [True, True, True, True]
_visibility_rgb: list = [True, True, True]

# ---------------------------------------------------------------------------
# Helper: display a scene in the napari viewer
# ---------------------------------------------------------------------------

def _lock_colormap(layer, cmap: str) -> None:
    """Snap a layer's colormap back to cmap if it was changed."""
    if layer.colormap.name != cmap:
        layer.colormap = cmap


def _channel_slice(arr: np.ndarray, channel_idx: int, z_idx: int) -> np.ndarray:
    """Return a 2D image from a channel-first or time/channel-first scene."""
    ch_ax = _channel_axis % arr.ndim
    indexer = [slice(None)] * arr.ndim
    indexer[ch_ax] = channel_idx

    # Supported shapes for now are (C, Z, Y, X), plus a future
    # (T, C, Z, Y, X) path where T is displayed at index 0.
    for ax in range(ch_ax):
        indexer[ax] = 0
    z_axis = ch_ax + 1 if ch_ax + 1 < arr.ndim else None
    if z_axis is not None and arr.shape[z_axis] > 1:
        indexer[z_axis] = min(z_idx, arr.shape[z_axis] - 1)
    elif z_axis is not None:
        indexer[z_axis] = 0

    return arr[tuple(indexer)]


def _z_count(arr: np.ndarray) -> int:
    ch_ax = _channel_axis % arr.ndim
    z_axis = ch_ax + 1
    if z_axis >= arr.ndim:
        return 1
    return arr.shape[z_axis]


def _bgr_to_rgb_for_save(arr: np.ndarray) -> np.ndarray:
    """Move converted BGR channels to RGB-last for TIFF writing."""
    rgb = np.take(arr, [2, 1, 0], axis=_channel_axis)
    return np.moveaxis(rgb, _channel_axis, -1)


def _remember_layer_visibility(viewer: napari.Viewer, view_mode: str) -> None:
    global _visibility_original, _visibility_rgb

    if view_mode == "rgb" and len(viewer.layers) == 3:
        _visibility_rgb = [viewer.layers[c].visible for c in range(3)]
    elif view_mode == "original" and len(viewer.layers) == len(_visibility_original):
        _visibility_original = [viewer.layers[c].visible for c in range(len(_visibility_original))]


def _display_scene(
    viewer: napari.Viewer,
    scene_idx: int,
    remember_visibility: bool = True,
) -> None:
    global _current_scene
    _current_scene = scene_idx

    if remember_visibility:
        _remember_layer_visibility(viewer, _view_mode)

    viewer.layers.clear()

    if _view_mode == "rgb" and _scenes_rgb:
        arr = _scenes_rgb[scene_idx]
        for c, (cmap, name) in enumerate(zip(COLORMAPS_RGB, NAMES_RGB)):
            layer = viewer.add_image(
                _channel_slice(arr, c, _current_z),
                name=name,
                colormap=cmap,
                blending="additive",
                visible=_visibility_rgb[c],
            )
            # Lock colormap: snap back if user tries to change it
            layer.events.colormap.connect(
                lambda e, _layer=layer, _cmap=cmap: _lock_colormap(_layer, _cmap)
            )
    else:
        arr = _scenes[scene_idx]
        n_channels = arr.shape[_channel_axis]
        for c in range(n_channels):
            cmap = COLORMAPS_ORIGINAL[c % len(COLORMAPS_ORIGINAL)]
            name = _channel_names[c] if c < len(_channel_names) else f"Ch {c}"
            viewer.add_image(
                _channel_slice(arr, c, _current_z),
                name=name,
                colormap=cmap,
                blending="additive",
                visible=_visibility_original[c] if c < len(_visibility_original) else True,
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
        scene_slider_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(34)
        self.prev_btn.setToolTip("Previous scene")
        self.prev_btn.clicked.connect(self._prev_scene)
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(34)
        self.next_btn.setToolTip("Next scene")
        self.next_btn.clicked.connect(self._next_scene)
        scene_slider_row.addWidget(self.prev_btn)
        scene_slider_row.addWidget(self.slider)
        scene_slider_row.addWidget(self.next_btn)
        nav_layout.addLayout(scene_slider_row)

        self.z_label = QLabel("Z 1 / ?")
        self.z_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.z_label)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setTickPosition(QSlider.TicksBelow)
        self.z_slider.valueChanged.connect(self._on_z_slider)
        z_slider_row = QHBoxLayout()
        self.z_prev_btn = QPushButton("◀")
        self.z_prev_btn.setFixedWidth(34)
        self.z_prev_btn.setToolTip("Previous Z")
        self.z_prev_btn.clicked.connect(self._prev_z)
        self.z_next_btn = QPushButton("▶")
        self.z_next_btn.setFixedWidth(34)
        self.z_next_btn.setToolTip("Next Z")
        self.z_next_btn.clicked.connect(self._next_z)
        z_slider_row.addWidget(self.z_prev_btn)
        z_slider_row.addWidget(self.z_slider)
        z_slider_row.addWidget(self.z_next_btn)
        nav_layout.addLayout(z_slider_row)
 
        self.nav_widget.setVisible(False)
        self.nav_dock = self.viewer.window.add_dock_widget(
            self.nav_widget,
            name="Scene / Z",
            area="bottom",
        )
        self.nav_dock.setVisible(False)
 
        # ── RGB Conversion group ───────────────────────────────────────
        self.rgb_group = QGroupBox("RGB Conversion")
        self.rgb_group.setVisible(False)
        rgb_layout = QVBoxLayout()
        rgb_layout.setSpacing(6)
        self.rgb_group.setLayout(rgb_layout)
 
        # convert_mode
        rgb_layout.addWidget(QLabel("Convert mode:"))
        self.combo_convert_mode = QComboBox()
        self.combo_convert_mode.addItems(["RemoveFarRed", "MergeRed", "MergeMagenta"])
        rgb_layout.addWidget(self.combo_convert_mode)

        # channel assignment
        rgb_layout.addWidget(QLabel("Channel assignment:"))
        assignment_grid = QGridLayout()
        self.channel_combos = {}
        for row, channel_key in enumerate(["far-red", "red", "green", "blue"]):
            label = QLabel(channel_key)
            combo = QComboBox()
            self.channel_combos[channel_key] = combo
            assignment_grid.addWidget(label, row, 0)
            assignment_grid.addWidget(combo, row, 1)
        rgb_layout.addLayout(assignment_grid)
 
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

        self.reset_rgb_btn = QPushButton("Reset RGB")
        self.reset_rgb_btn.setEnabled(False)
        self.reset_rgb_btn.clicked.connect(self._reset_rgb)
        rgb_layout.addWidget(self.reset_rgb_btn)
 
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
        global _scenes, _metadata, _channel_names, _scene_keys, _scenes_rgb
        global _view_mode, _visibility_original, _visibility_rgb, _conversion_params
        global _channel_axis, _current_z
 
        path = self.path_edit.text().strip()
        self.status_label.setText("Loading…")
        self.load_btn.setEnabled(False)
        self.repaint()
 
        try:
            _scenes, _metadata, _channel_axis = cp.load_czi(path)
        except Exception as e:
            traceback.print_exc()
            self.status_label.setText(f"Error: {e}")
            self.load_btn.setEnabled(True)
            return
 
        # Extract channel names from metadata
        try:
            channels = _metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]
            _channel_names = [ch["@Name"] for ch in channels]
        except Exception:
            sample = next(iter(_scenes.values()))
            _channel_names = [f"Ch {c}" for c in range(sample.shape[_channel_axis])]

        _scene_keys = sorted(_scenes.keys())
        _conversion_params = {"channel_axis": _channel_axis}
        self._populate_channel_assignment_controls()
 
        # Reset RGB state
        _scenes_rgb = {}
        _visibility_original = [True] * len(_channel_names)
        _visibility_rgb = [True, True, True]
        _view_mode = "original"
        _current_z = 0
        self.radio_original.setChecked(True)
        self.radio_original.setEnabled(False)
        self.radio_rgb.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.reset_rgb_btn.setEnabled(False)
        self.convert_status.setText("")
 
        n = len(_scenes)
        self.status_label.setText(f"Loaded {n} scene(s).")
 
        self.slider.setMaximum(n - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(max(1, n // 10))
        self._update_nav_label(0, n)
        self._set_z_controls_for_scene(_scene_keys[0])
        self.nav_widget.setVisible(True)
        self.nav_dock.setVisible(True)
        self.rgb_group.setVisible(True)
        self.load_btn.setEnabled(True)
 
        _display_scene(self.viewer, _scene_keys[0])
        self._update_arrow_states(0, n)
 
    # ------------------------------------------------------------------
    # Slots — scene navigation
    # ------------------------------------------------------------------
    def _on_slider(self, value: int):
        n = len(_scenes)
        if n == 0:
            return
        self._update_nav_label(value, n)
        scene_key = _scene_keys[value]
        self._set_z_controls_for_scene(scene_key)
        _display_scene(self.viewer, scene_key)
        self._update_arrow_states(value, n)

    def _on_z_slider(self, value: int):
        global _current_z
        if not _scenes:
            return
        _current_z = value
        self._update_z_label(value, self.z_slider.maximum() + 1)
        self._update_z_arrow_states(value, self.z_slider.maximum() + 1)
        _display_scene(self.viewer, _scene_keys[self.slider.value()])
 
    def _prev_scene(self):
        v = self.slider.value()
        if v > 0:
            self.slider.setValue(v - 1)
 
    def _next_scene(self):
        v = self.slider.value()
        if v < self.slider.maximum():
            self.slider.setValue(v + 1)

    def _prev_z(self):
        v = self.z_slider.value()
        if v > 0:
            self.z_slider.setValue(v - 1)

    def _next_z(self):
        v = self.z_slider.value()
        if v < self.z_slider.maximum():
            self.z_slider.setValue(v + 1)
 
    # ------------------------------------------------------------------
    # Slots — RGB conversion
    # ------------------------------------------------------------------
    def _on_norm_mode_changed(self, text: str):
        is_pctile = text == "Percentile"
        self.pctile_spin.setEnabled(is_pctile)
        self.pctile_label.setEnabled(is_pctile)
 
    def _convert(self):
        global _scenes_rgb, _view_mode, _last_conversion_params, _conversion_params

        self.convert_status.setText("Converting…")
        self.convert_btn.setEnabled(False)
        self.repaint()

        channel_assignment = self._channel_assignment()
 
        params = {
            "convert_mode":        self.combo_convert_mode.currentText(),
            "norm_mode":           self.combo_norm_mode.currentText(),
            "pctile_value":        self.pctile_spin.value() if self.combo_norm_mode.currentText() == "Percentile" else None,
            "norm_before_combine": self.combo_norm_before.currentText() == "True",
            "norm_after_combine":  self.combo_norm_after.currentText() == "True",
            "channel_axis":        _channel_axis,
            "channel_assignment":  channel_assignment,
        }
        processing_params = dict(params)
        processing_params["channel_assignment"] = {
            **channel_assignment,
            "far_red": channel_assignment["far-red"],
        }
 
        try:
            _scenes_rgb = cp.convert_to_rgb_all_scenes(_scenes, processing_params)
            _conversion_params = dict(params)
            _last_conversion_params = params
        except Exception as e:
            traceback.print_exc()
            self.convert_status.setText(f"Error: {e}")
            self.convert_btn.setEnabled(True)
            return
 
        self.convert_status.setText("Conversion done.")
        self.convert_btn.setEnabled(True)
        self.radio_original.setEnabled(True)
        self.radio_rgb.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.reset_rgb_btn.setEnabled(True)
 
        # Auto-switch view to RGB after conversion
        self.radio_rgb.setChecked(True)
        if _view_mode != "rgb":
            _view_mode = "rgb"
            _display_scene(self.viewer, _current_scene, remember_visibility=False)
 
    def _on_view_toggle(self):
        global _view_mode
        if not _scenes:
            return

        next_view_mode = "original" if self.radio_original.isChecked() else "rgb"
        if next_view_mode == "rgb" and not _scenes_rgb:
            return
        if next_view_mode == _view_mode:
            return

        _remember_layer_visibility(self.viewer, _view_mode)
        _view_mode = next_view_mode
        _display_scene(self.viewer, _current_scene, remember_visibility=False)

    def _reset_rgb(self):
        global _scenes_rgb, _view_mode, _visibility_rgb
        global _conversion_params, _last_conversion_params

        if not _scenes_rgb:
            return

        _remember_layer_visibility(self.viewer, _view_mode)
        _scenes_rgb = {}
        _visibility_rgb = [True, True, True]
        _conversion_params = {"channel_axis": _channel_axis}
        _last_conversion_params = {}
        self.save_btn.setEnabled(False)
        self.reset_rgb_btn.setEnabled(False)
        self.radio_rgb.setEnabled(False)
        self.radio_original.setEnabled(False)
        _view_mode = "original"
        self.radio_original.setChecked(True)
        self.convert_status.setText("RGB reset.")
        _display_scene(self.viewer, _current_scene, remember_visibility=False)
 
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
                    arr_rgb = _bgr_to_rgb_for_save(_scenes_rgb[k])
                    axes_by_ndim = {3: "YXS", 4: "ZYXS", 5: "TZYXS"}
                    axes = axes_by_ndim.get(arr_rgb.ndim)
                    metadata = {"axes": axes} if axes else None
                    tif.write(
                        arr_rgb,
                        contiguous=False,
                        photometric="rgb",
                        metadata=metadata,
                    )
 
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
            traceback.print_exc()
            self.convert_status.setText(f"Save error: {e}")
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_nav_label(self, idx: int, total: int):
        self.scene_label.setText(f"Scene {idx + 1} / {total}")

    def _update_z_label(self, idx: int, total: int):
        self.z_label.setText(f"Z {idx + 1} / {total}")
 
    def _update_arrow_states(self, idx: int, total: int):
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < total - 1)

    def _update_z_arrow_states(self, idx: int, total: int):
        has_multiple_z = total > 1
        self.z_prev_btn.setEnabled(has_multiple_z and idx > 0)
        self.z_next_btn.setEnabled(has_multiple_z and idx < total - 1)

    def _set_z_controls_for_scene(self, scene_key: int):
        global _current_z
        total_z = _z_count(_scenes[scene_key])
        next_z = min(_current_z, total_z - 1)
        self.z_slider.blockSignals(True)
        self.z_slider.setMaximum(total_z - 1)
        self.z_slider.setValue(next_z)
        self.z_slider.setTickInterval(max(1, total_z // 10))
        self.z_slider.setEnabled(total_z > 1)
        self.z_slider.blockSignals(False)
        _current_z = next_z
        self._update_z_label(next_z, total_z)
        self._update_z_arrow_states(next_z, total_z)

    def _populate_channel_assignment_controls(self):
        defaults = {"far-red": 3, "red": 2, "green": 1, "blue": 0}
        for channel_key, combo in self.channel_combos.items():
            combo.blockSignals(True)
            combo.clear()
            for idx, name in enumerate(_channel_names):
                combo.addItem(f"{idx}: {name}", idx)
            default_idx = min(defaults[channel_key], max(0, len(_channel_names) - 1))
            combo.setCurrentIndex(default_idx)
            combo.blockSignals(False)

    def _channel_assignment(self) -> dict:
        return {
            channel_key: combo.currentData()
            for channel_key, combo in self.channel_combos.items()
        }
 
 
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
 
