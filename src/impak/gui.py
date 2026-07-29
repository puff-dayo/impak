from __future__ import annotations

import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import impak

try:
    from PySide6.QtCore import (
        Qt, QThread, Signal, QObject, Slot
    )
    from PySide6.QtGui import QTextCursor, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
        QDoubleSpinBox, QFileDialog, QTextEdit, QProgressBar,
        QSplitter, QListWidget, QMessageBox, QFormLayout,
        QTabWidget, QScrollArea
    )

    _HAS_GUI = True
except ImportError:
    _HAS_GUI = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _natural_sort_key(path: Path) -> list:
    parts = []
    for chunk in re.split(r"(\d+)", path.name):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


def collect_images_from_folder(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=_natural_sort_key,
    )


def collect_images_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return sorted(
        (p for p in extract_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS),
        key=_natural_sort_key,
    )


# Worker thread

class EncodeWorker(QObject):
    progress = Signal(int, int, str)  # current, total, message
    log = Signal(str)
    finished = Signal(bool, str)  # success, message

    def __init__(
            self,
            image_paths: list[Path],
            output_path: Path,
            params: dict,
            baseline_paths: list[Path],
    ):
        super().__init__()
        self._image_paths = image_paths
        self._output_path = output_path
        self._params = params
        self._baseline_paths = baseline_paths
        self._cancelled = False

    @Slot()
    def run(self):
        try:
            params = dict(self._params)
            if self._baseline_paths:
                params["baselines"] = self._baseline_paths

            total = len(self._image_paths)
            self.log.emit(f"Starting encode: {total} images → {self._output_path}")
            self.log.emit(f"Mode: {params['mode']}  Codec: {params['codec']}  Quality: {params['quality']}")

            with impak.create(self._output_path, **params) as w:
                for i, p in enumerate(self._image_paths):
                    if self._cancelled:
                        self.finished.emit(False, "Cancelled by user.")
                        return
                    self.progress.emit(i, total, p.name)
                    self.log.emit(f"  [{i + 1:>4}/{total}] {p.name}")
                    w.add(p, name=p.stem)

            size_kb = self._output_path.stat().st_size / 1024
            msg = f"Done. Output: {self._output_path}  ({size_kb:.1f} KB)"
            self.log.emit(msg)
            self.finished.emit(True, msg)

        except Exception as exc:
            self.finished.emit(False, f"Error: {exc}")

    def cancel(self):
        self._cancelled = True


# Main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"impak Encoder v{impak.__version__}")
        self.setMinimumSize(860, 768)

        self._worker: Optional[EncodeWorker] = None
        self._thread: Optional[QThread] = None
        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._image_paths: list[Path] = []
        self._baseline_paths: list[Path] = []

        self._build_ui()

    # UI construction

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel – settings
        left = QWidget()
        left.setMinimumWidth(340)
        left.setMaximumWidth(420)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        lv.addWidget(self._build_input_group())
        lv.addWidget(self._build_output_group())
        lv.addWidget(self._build_mode_group())
        lv.addWidget(self._build_codec_group())
        lv.addWidget(self._build_advanced_group())
        lv.addWidget(self._build_baseline_group())
        lv.addStretch()

        # Bottom buttons on left panel
        btn_row = QHBoxLayout()
        self._btn_encode = QPushButton("Encode")
        self._btn_encode.setFixedHeight(34)
        self._btn_encode.clicked.connect(self._start_encode)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedHeight(34)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_encode)
        btn_row.addWidget(self._btn_encode)
        btn_row.addWidget(self._btn_cancel)
        lv.addLayout(btn_row)

        splitter.addWidget(left)

        # Right panel – tabs, Images + Preview and log
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)

        tabs = QTabWidget()
        rv.addWidget(tabs)

        images_tab = QWidget()
        images_layout = QVBoxLayout(images_tab)
        images_layout.setContentsMargins(4, 4, 4, 4)

        img_splitter = QSplitter(Qt.Vertical)
        images_layout.addWidget(img_splitter)

        self._image_list = QListWidget()
        self._image_list.setAlternatingRowColors(True)
        self._image_list.currentRowChanged.connect(self._on_image_selected)
        img_splitter.addWidget(self._image_list)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 4, 0, 0)

        preview_header = QLabel("Preview")
        preview_header.setAlignment(Qt.AlignCenter)
        preview_header.setStyleSheet("font-weight: bold; color: palette(mid);")
        preview_layout.addWidget(preview_header)

        self._preview_scroll = QScrollArea()
        self._preview_scroll.setAlignment(Qt.AlignCenter)
        self._preview_scroll.setWidgetResizable(False)
        self._preview_scroll.setMinimumHeight(160)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setText("Select an image to preview")
        self._preview_label.setStyleSheet("color: palette(mid); font-style: italic;")
        self._preview_label.setMinimumSize(100, 100)
        self._preview_scroll.setWidget(self._preview_label)
        preview_layout.addWidget(self._preview_scroll)

        self._preview_info = QLabel("")
        self._preview_info.setAlignment(Qt.AlignCenter)
        self._preview_info.setStyleSheet("color: palette(mid); font-size: 10px;")
        preview_layout.addWidget(self._preview_info)

        img_splitter.addWidget(preview_container)
        img_splitter.setSizes([300, 240])

        tabs.addTab(images_tab, "Images")

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QTextEdit.WidgetWidth)
        tabs.addTab(self._log, "Log")

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        rv.addWidget(self._progress)

        self._status = QLabel("Ready.")
        rv.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setSizes([380, 480])

    def _build_input_group(self) -> QGroupBox:
        g = QGroupBox("Input")
        fl = QFormLayout(g)
        fl.setLabelAlignment(Qt.AlignRight)

        row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Folder or .zip file…")
        self._input_edit.setReadOnly(True)
        btn_folder = QPushButton("Folder…")
        btn_folder.clicked.connect(self._pick_folder)
        btn_zip = QPushButton("ZIP…")
        btn_zip.clicked.connect(self._pick_zip)
        row.addWidget(self._input_edit)
        row.addWidget(btn_folder)
        row.addWidget(btn_zip)
        fl.addRow("Source:", row)

        self._img_count_label = QLabel("No images loaded.")
        fl.addRow("", self._img_count_label)
        return g

    def _build_output_group(self) -> QGroupBox:
        g = QGroupBox("Output")
        fl = QFormLayout(g)
        fl.setLabelAlignment(Qt.AlignRight)

        row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("output.impak")
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._pick_output)
        row.addWidget(self._output_edit)
        row.addWidget(btn)
        fl.addRow("File:", row)
        return g

    def _build_mode_group(self) -> QGroupBox:
        g = QGroupBox("Diff Mode")
        fl = QFormLayout(g)
        fl.setLabelAlignment(Qt.AlignRight)

        self._mode_combo = QComboBox()
        modes = [
            ("lto", "LTO – best reference (recommended)"),
            ("vs_first", "vs_first – diff against frame 0"),
            ("vs_prior", "vs_prior – diff against previous"),
            ("keyframe", "keyframe – periodic full frames"),
            ("manual", "manual – pinned baseline images"),
        ]
        for value, label in modes:
            self._mode_combo.addItem(label, userData=value)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_change)
        fl.addRow("Mode:", self._mode_combo)

        self._kf_interval_spin = QSpinBox()
        self._kf_interval_spin.setRange(1, 9999)
        self._kf_interval_spin.setValue(10)
        self._kf_interval_spin.setEnabled(False)
        self._kf_interval_row_label = QLabel("Keyframe every:")
        self._kf_interval_row_label.setEnabled(False)
        fl.addRow(self._kf_interval_row_label, self._kf_interval_spin)

        return g

    def _build_codec_group(self) -> QGroupBox:
        g = QGroupBox("Codec")
        fl = QFormLayout(g)
        fl.setLabelAlignment(Qt.AlignRight)

        self._codec_combo = QComboBox()
        self._codec_combo.addItem("WebP (default, smaller)", userData="webp")
        self._codec_combo.addItem("PNG (lossless)", userData="png")
        self._codec_combo.currentIndexChanged.connect(self._on_codec_change)
        fl.addRow("Codec:", self._codec_combo)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(0, 100)
        self._quality_spin.setValue(95)
        self._quality_spin.setSuffix("")
        self._quality_spin.valueChanged.connect(self._on_quality_change)
        self._quality_label = QLabel("Quality:")
        fl.addRow(self._quality_label, self._quality_spin)

        return g

    def _build_advanced_group(self) -> QGroupBox:
        g = QGroupBox("Advanced")
        g.setCheckable(True)
        g.setChecked(False)
        self._advanced_group = g
        fl = QFormLayout(g)
        fl.setLabelAlignment(Qt.AlignRight)

        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 255)
        self._threshold_spin.setValue(4)
        self._threshold_spin.setToolTip("Per-channel pixel delta treated as unchanged. 0 = perfectly lossless.")
        fl.addRow("Threshold:", self._threshold_spin)

        self._tile_size_spin = QSpinBox()
        self._tile_size_spin.setRange(4, 256)
        self._tile_size_spin.setSingleStep(4)
        self._tile_size_spin.setValue(64)
        self._tile_size_spin.setToolTip("Diff grid tile size in pixels (16–64 typical).")
        fl.addRow("Tile size:", self._tile_size_spin)

        self._merge_gap_spin = QSpinBox()
        self._merge_gap_spin.setRange(0, 256)
        self._merge_gap_spin.setValue(8)
        self._merge_gap_spin.setToolTip("Merge changed tiles within this many pixels.")
        fl.addRow("Merge gap:", self._merge_gap_spin)

        self._auto_kf_sim_spin = QDoubleSpinBox()
        self._auto_kf_sim_spin.setRange(0.0, 1.0)
        self._auto_kf_sim_spin.setSingleStep(0.05)
        self._auto_kf_sim_spin.setValue(0.5)
        self._auto_kf_sim_spin.setDecimals(2)
        self._auto_kf_sim_spin.setToolTip("Force keyframe when similarity drops below this. 0 = never.")
        fl.addRow("Auto KF sim:", self._auto_kf_sim_spin)

        self._lto_candidates_spin = QSpinBox()
        self._lto_candidates_spin.setRange(1, 64)
        self._lto_candidates_spin.setValue(6)
        self._lto_candidates_spin.setToolTip("(LTO/manual) Number of top-similarity frames to fully probe.")
        fl.addRow("LTO candidates:", self._lto_candidates_spin)

        self._max_ref_depth_spin = QSpinBox()
        self._max_ref_depth_spin.setRange(1, 128)
        self._max_ref_depth_spin.setValue(8)
        self._max_ref_depth_spin.setToolTip("(LTO) Max decode-chain depth before forcing a keyframe.")
        fl.addRow("Max ref depth:", self._max_ref_depth_spin)

        return g

    def _build_baseline_group(self) -> QGroupBox:
        g = QGroupBox("Baselines  (manual mode only)")
        self._baseline_group = g
        g.setEnabled(False)
        v = QVBoxLayout(g)

        self._baseline_list = QListWidget()
        self._baseline_list.setMaximumHeight(80)
        v.addWidget(self._baseline_list)

        row = QHBoxLayout()
        btn_add = QPushButton("Add…")
        btn_add.clicked.connect(self._add_baselines)
        btn_rem = QPushButton("Remove")
        btn_rem.clicked.connect(self._remove_baseline)

        self._fallback_combo = QComboBox()
        for val, lbl in [("lto", "lto"), ("vs_prior", "vs_prior"), ("vs_first", "vs_first"), ("keyframe", "keyframe")]:
            self._fallback_combo.addItem(lbl, userData=val)
        row.addWidget(btn_add)
        row.addWidget(btn_rem)
        row.addStretch()
        row.addWidget(QLabel("Fallback:"))
        row.addWidget(self._fallback_combo)
        v.addLayout(row)
        return g

    # Slots

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self._load_from_folder(Path(folder))

    def _pick_zip(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ZIP file", "", "ZIP files (*.zip)"
        )
        if not path:
            return
        self._load_from_zip(Path(path))

    def _pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save .impak file", "", "impak files (*.impak)"
        )
        if path:
            if not path.endswith(".impak"):
                path += ".impak"
            self._output_edit.setText(path)

    def _load_from_folder(self, folder: Path):
        paths = collect_images_from_folder(folder)
        if not paths:
            QMessageBox.warning(self, "No images", f"No image files found in:\n{folder}")
            return
        self._image_paths = paths
        self._input_edit.setText(str(folder))
        self._refresh_image_list()
        self._output_edit.setText(str(folder.parent / (folder.name + ".impak")))

    def _load_from_zip(self, zip_path: Path):
        if self._tmp_dir:
            self._tmp_dir.cleanup()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="impak_gui_")
        tmp = Path(self._tmp_dir.name)
        try:
            paths = collect_images_from_zip(zip_path, tmp)
        except Exception as exc:
            QMessageBox.critical(self, "ZIP error", str(exc))
            return
        if not paths:
            QMessageBox.warning(self, "No images", f"No image files found in:\n{zip_path}")
            return
        self._image_paths = paths
        self._input_edit.setText(str(zip_path))
        self._refresh_image_list()
        self._output_edit.setText(str(zip_path.with_suffix(".impak")))

    def _refresh_image_list(self):
        self._image_list.clear()
        for p in self._image_paths:
            self._image_list.addItem(p.name)
        n = len(self._image_paths)
        self._img_count_label.setText(f"{n} image{'s' if n != 1 else ''} found.")
        self._preview_label.setText("Select an image to preview")
        self._preview_label.setPixmap(QPixmap())  # clear any previous pixmap
        self._preview_info.setText("")

    def _on_image_selected(self, row: int):
        if row < 0 or row >= len(self._image_paths):
            self._preview_label.setText("Select an image to preview")
            self._preview_label.setPixmap(QPixmap())
            self._preview_info.setText("")
            return

        path = self._image_paths[row]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._preview_label.setText(f"Cannot load: {path.name}")
            self._preview_info.setText("")
            return

        m = self._preview_scroll.contentsMargins()
        max_w = max(self._preview_scroll.width() - m.left() - m.right() - 4, 120)
        max_h = max(self._preview_scroll.height() - m.top() - m.bottom() - 8, 100)
        scaled = pixmap.scaled(
            max_w, max_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
        self._preview_label.resize(scaled.size())
        self._preview_info.setText(
            f"{path.name}  ·  {pixmap.width()} × {pixmap.height()} px"
        )

    def _on_mode_change(self, _idx):
        mode = self._mode_combo.currentData()
        is_manual = mode == "manual"
        is_keyframe = mode == "keyframe"
        self._baseline_group.setEnabled(is_manual)
        self._kf_interval_spin.setEnabled(is_keyframe)
        self._kf_interval_row_label.setEnabled(is_keyframe)

    def _on_codec_change(self, _idx):
        codec = self._codec_combo.currentData()
        is_png = codec == "png"
        self._quality_spin.setEnabled(not is_png)
        self._quality_label.setEnabled(not is_png)
        if is_png:
            self._quality_spin.setValue(100)

    def _on_quality_change(self, value: int):
        if value == 100:
            self._quality_spin.setSuffix("  (near lossless)")
        else:
            self._quality_spin.setSuffix(f"  (much lossy)")

    def _add_baselines(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select baseline images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        for p in paths:
            pp = Path(p)
            if pp not in self._baseline_paths:
                self._baseline_paths.append(pp)
                self._baseline_list.addItem(pp.name)

    def _remove_baseline(self):
        row = self._baseline_list.currentRow()
        if row >= 0:
            self._baseline_list.takeItem(row)
            self._baseline_paths.pop(row)

    # Encode

    def _start_encode(self):
        if not self._image_paths:
            QMessageBox.warning(self, "No input", "Please select a folder or ZIP file first.")
            return
        out = self._output_edit.text().strip()
        if not out:
            QMessageBox.warning(self, "No output", "Please specify an output .impak path.")
            return

        mode = self._mode_combo.currentData()
        if mode == "manual" and not self._baseline_paths:
            QMessageBox.warning(self, "Baselines required",
                                "Manual mode requires at least one baseline image.")
            return

        params = dict(
            mode=mode,
            codec=self._codec_combo.currentData(),
            quality=self._quality_spin.value(),
            threshold=self._threshold_spin.value(),
            tile_size=self._tile_size_spin.value(),
            merge_gap=self._merge_gap_spin.value(),
            auto_keyframe_sim=self._auto_kf_sim_spin.value(),
            lto_candidates=self._lto_candidates_spin.value(),
            max_ref_depth=self._max_ref_depth_spin.value(),
        )
        if mode == "keyframe":
            params["keyframe_interval"] = self._kf_interval_spin.value()
        if mode == "manual":
            params["fallback_mode"] = self._fallback_combo.currentData()

        self._log.clear()
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._image_paths))
        self._progress.setValue(0)
        self._btn_encode.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._status.setText("Encoding…")

        self._thread = QThread(self)
        self._worker = EncodeWorker(
            self._image_paths, Path(out), params, list(self._baseline_paths)
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _cancel_encode(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._status.setText("Cancelling…")

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, name: str):
        self._progress.setValue(current + 1)
        self._status.setText(f"Encoding {current + 1}/{total}: {name}")

    @Slot(str)
    def _on_log(self, msg: str):
        self._log.append(msg)
        self._log.moveCursor(QTextCursor.End)

    @Slot(bool, str)
    def _on_finished(self, success: bool, msg: str):
        self._btn_encode.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText(msg)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

        if success:
            QMessageBox.information(self, "Encode complete", msg)
        else:
            QMessageBox.critical(self, "Encode failed", msg)

    def closeEvent(self, event):
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        if self._tmp_dir:
            self._tmp_dir.cleanup()
        super().closeEvent(event)


def main():
    if not _HAS_GUI:
        print("impak[gui] is not installed.  Run:  pip install 'impak[gui]'", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("impak Encoder")
    app.setApplicationVersion(impak.__version__)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
