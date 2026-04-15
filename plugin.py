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
import os, sys

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel,
    QSlider, QFileDialog, QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator

from pylibCZIrw import czi as pyczi
import json
import czi_processing as cp

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_scenes: dict = {}       # {scene_index: np.ndarray (Y, X, 4)}
_metadata: dict = {}     # raw metadata dict from backend
_current_scene: int = 0
_channel_names: list = []
# _channel_names = cp.get_channel_names(_metadata)

COLORMAPS = ["blue", "green", "red", "magenta"]   # one per channel (C=4)


# ---------------------------------------------------------------------------
# Helper: display a scene in the napari viewer
# ---------------------------------------------------------------------------
def _display_scene(viewer: napari.Viewer, scene_idx: int) -> None:
    """Clear existing layers and display all 4 channels for scene_idx."""
    global _current_scene
    _current_scene = scene_idx

    viewer.layers.clear()

    arr = _scenes[scene_idx]   # (Y, X, 4)
    for c, cmap in enumerate(COLORMAPS):
        viewer.add_image(
            arr[:, :, c],
            name=_channel_names[c],
            colormap=cmap,
            blending="additive",
            visible=True,
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

        # ── Scene navigation (hidden until a file is loaded) ───────────
        self.nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(6)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_widget.setLayout(nav_layout)

        # Scene label  "Scene 1 / 11"
        self.scene_label = QLabel("Scene 1 / ?")
        self.scene_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.scene_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider)
        nav_layout.addWidget(self.slider)

        # Prev / Next arrows
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

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
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
        global _scenes, _metadata, _channel_names

        path = self.path_edit.text().strip()
        self.status_label.setText("Loading…")
        self.load_btn.setEnabled(False)
        self.repaint()   # force UI refresh so "Loading…" shows immediately

        try:
            _scenes, _metadata = cp.load_czi(path)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.load_btn.setEnabled(True)
            return

        _channel_names = cp.get_channel_names(_metadata)
        n = len(_scenes)
        self.status_label.setText(f"Loaded {n} scene(s).")

        # Configure slider
        self.slider.setMaximum(n - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(max(1, n // 10))

        self._update_nav_label(0, n)
        self.nav_widget.setVisible(True)
        self.load_btn.setEnabled(True)

        # Display first scene
        _display_scene(self.viewer, 0)
        self._update_arrow_states(0, n)

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
            self.slider.setValue(v - 1)   # triggers _on_slider

    def _next_scene(self):
        v = self.slider.value()
        if v < self.slider.maximum():
            self.slider.setValue(v + 1)   # triggers _on_slider

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