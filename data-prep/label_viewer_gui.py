"""Review and fix labels by hand (`make label`).

Keys: S = spam, H = ham (both move on to the next message), Up/Down to browse,
Ctrl+S to save. Shift/Ctrl+click or Ctrl+A select several messages; S/H then label
them all at once. Qt lays out Persian (RTL) text itself, line by line.
"""
import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from spam_common.data import LABELED_FILE, write_jsonl
from spam_common.model_label import SCORE_FIELD, UNCERTAIN

LABEL_FILTERS = ["All", "spam", "ham", "null"]
REGEX_FIELDS = ["text", "address", "both"]
SCORE_FILTERS = ["All", f"Uncertain ({UNCERTAIN[0]}-{UNCERTAIN[1]})", "Labelled by model", "From dataset"]
SORT_ORDERS = ["File order", "Most uncertain first"]
COLUMNS = ["ID", "Label", "Spam score", "Address", "Text preview"]
PREVIEW_CHARS = 60


def filter_indices(data, label_filter="All", pattern="", field="text", score_filter=0, sort=0):
    """Indices of the rows in `data` that pass the filters, in display order.

    score_filter and sort are indices into SCORE_FILTERS and SORT_ORDERS.
    Raises re.error for an invalid pattern.
    """
    regex = re.compile(pattern, re.IGNORECASE) if pattern else None
    indices = []
    for i, item in enumerate(data):
        # "null" matches rows whose label is missing, null or empty.
        if label_filter != "All" and (item.get("label") or "null") != label_filter:
            continue

        score = item.get(SCORE_FIELD)
        if score_filter == 1 and not (score is not None and UNCERTAIN[0] <= score <= UNCERTAIN[1]):
            continue
        if score_filter == 2 and score is None:
            continue
        if score_filter == 3 and (SCORE_FIELD not in item or score is not None):
            continue

        if regex and not (
            (field in ("address", "both") and regex.search(str(item.get("address", ""))))
            or (field in ("text", "both") and regex.search(str(item.get("text", ""))))
        ):
            continue

        indices.append(i)

    if sort == 1:
        # Closest to 0.5 first; rows without a score last.
        def uncertainty(i):
            score = data[i].get(SCORE_FIELD)
            return abs(score - 0.5) if score is not None else 1.0
        indices.sort(key=uncertainty)
    return indices


def read_jsonl_lenient(path):
    """Rows of a JSONL file, plus the line numbers that are not valid JSON."""
    rows, bad_lines = [], []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines.append(line_num)
    return rows, bad_lines


def row_values(i, item):
    score = item.get(SCORE_FIELD)
    text = " ".join(item.get("text", "").split())
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "…"
    return [
        str(i),
        item.get("label") or "null",
        "—" if score is None else f"{score:.2f}",
        str(item.get("address", "")),
        text,
    ]


class SMSLabeler(QMainWindow):
    def __init__(self, path=str(LABELED_FILE)):
        super().__init__()
        self.data = []
        self.filtered = []  # indices into self.data, one per table row
        self.filepath = ""
        self.resize(1000, 750)
        self.setup_ui(path)
        self.update_title()

    def setup_ui(self, path):
        # --- Controls & filters ---
        controls = QGroupBox("Controls && Filters")
        controls_layout = QVBoxLayout(controls)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.file_edit = QLineEdit(path)
        self.file_edit.returnPressed.connect(self.load_from_field)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(button("Browse", self.browse_file))
        file_row.addWidget(button("Load", self.load_from_field))
        controls_layout.addLayout(file_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Label:"))
        self.label_combo = combo(LABEL_FILTERS, self.apply_filters)
        filter_row.addWidget(self.label_combo)
        filter_row.addSpacing(15)
        filter_row.addWidget(QLabel("Regex:"))
        self.regex_edit = QLineEdit()
        self.regex_edit.setPlaceholderText("Enter to apply")
        self.regex_edit.returnPressed.connect(self.apply_filters)
        filter_row.addWidget(self.regex_edit, 1)
        filter_row.addWidget(QLabel("Field:"))
        self.field_combo = combo(REGEX_FIELDS, self.apply_filters)
        filter_row.addWidget(self.field_combo)
        filter_row.addWidget(button("Apply Filter", self.apply_filters))
        controls_layout.addLayout(filter_row)

        # Model score (written by `make model-label`): review uncertain messages first.
        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("Model score:"))
        self.score_combo = combo(SCORE_FILTERS, self.apply_filters)
        score_row.addWidget(self.score_combo)
        score_row.addSpacing(15)
        score_row.addWidget(QLabel("Sort:"))
        self.sort_combo = combo(SORT_ORDERS, self.apply_filters)
        score_row.addWidget(self.sort_combo)
        score_row.addStretch(1)
        controls_layout.addLayout(score_row)

        # --- Message table ---
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        for col, width in enumerate([60, 70, 90, 150]):
            self.table.setColumnWidth(col, width)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.show_current)

        # --- Entry details ---
        details = QGroupBox("Entry Details")
        details_layout = QVBoxLayout(details)
        detail_row = QHBoxLayout()
        self.index_label = QLabel("Index: -")
        self.position_label = QLabel("")
        self.position_label.setStyleSheet("color: gray")
        self.current_label = QLabel("")
        detail_row.addWidget(self.index_label)
        detail_row.addWidget(self.position_label)
        detail_row.addSpacing(20)
        detail_row.addWidget(QLabel("Label:"))
        detail_row.addWidget(self.current_label)
        detail_row.addSpacing(20)
        detail_row.addWidget(button("Spam (S)", lambda: self.set_label("spam")))
        detail_row.addWidget(button("Ham (H)", lambda: self.set_label("ham")))
        detail_row.addStretch(1)
        detail_row.addWidget(button("◄ Previous", lambda: self.go(-1)))
        detail_row.addWidget(button("Next ►", lambda: self.go(1)))
        detail_row.addSpacing(10)
        detail_row.addWidget(button("Save (Ctrl+S)", self.save_data))
        details_layout.addLayout(detail_row)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        font = self.detail_text.font()
        font.setPointSizeF(font.pointSizeF() + 3)
        self.detail_text.setFont(font)
        details_layout.addWidget(self.detail_text)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(controls)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # S/H only while the table or the details have focus, so typing a regex is safe.
        for key, label in (("S", "spam"), ("H", "ham")):
            shortcut = QShortcut(QKeySequence(key), splitter)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda label=label: self.set_label(label))
        QShortcut(QKeySequence.Save, self).activated.connect(self.save_data)

    # --- File handling ---

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSONL File", self.file_edit.text(), "JSONL files (*.jsonl);;All files (*)")
        if path:
            self.file_edit.setText(path)
            self.load_from_field()

    def load_from_field(self):
        if self.confirm_discard():
            self.load_data(self.file_edit.text())

    def load_data(self, path):
        try:
            data, bad_lines = read_jsonl_lenient(path)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found:\n{path}")
            return
        except (OSError, UnicodeDecodeError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            return

        self.data, self.filepath = data, path
        self.file_edit.setText(path)
        self.setWindowModified(False)
        self.update_title()
        self.apply_filters()
        self.statusBar().showMessage(f"Loaded {len(self.data)} entries.")
        if bad_lines:
            shown = ", ".join(map(str, bad_lines[:20])) + (" …" if len(bad_lines) > 20 else "")
            QMessageBox.warning(self, "Parse Error",
                                f"Skipped {len(bad_lines)} invalid line(s): {shown}\n"
                                "Saving will drop them from the file.")

    def save_data(self):
        if not self.filepath:
            QMessageBox.warning(self, "Warning", "No file is currently loaded.")
            return False
        try:
            write_jsonl(self.data, self.filepath)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            return False
        self.setWindowModified(False)
        self.statusBar().showMessage(f"Saved {len(self.data)} entries to {self.filepath}")
        return True

    def confirm_discard(self):
        """True if it is fine to drop the current data (saved, or the user chose to)."""
        if not self.isWindowModified():
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", "Save your label changes first?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Save:
            return self.save_data()
        return answer == QMessageBox.Discard

    def closeEvent(self, event):
        if self.confirm_discard():
            event.accept()
        else:
            event.ignore()

    def update_title(self):
        name = Path(self.filepath).name if self.filepath else "no file"
        self.setWindowTitle(f"SMS Labeler — {name}[*]")

    # --- Filtering and the table ---

    def apply_filters(self):
        try:
            self.filtered = filter_indices(
                self.data,
                self.label_combo.currentText(),
                self.regex_edit.text(),
                self.field_combo.currentText(),
                self.score_combo.currentIndex(),
                self.sort_combo.currentIndex(),
            )
        except re.error as e:
            QMessageBox.critical(self, "Regex Error", f"Invalid regular expression: {e}")
            return

        self.table.blockSignals(True)
        self.table.setRowCount(len(self.filtered))
        for row, i in enumerate(self.filtered):
            for col, value in enumerate(row_values(i, self.data[i])):
                cell = QTableWidgetItem(value)
                if col < 3:
                    cell.setTextAlignment(Qt.AlignCenter)
                elif col == 4:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, cell)
        self.table.blockSignals(False)

        if self.filtered:
            self.select_row(0)
        else:
            self.show_current()
        self.table.setFocus()

    def select_row(self, row):
        if 0 <= row < len(self.filtered):
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))

    def go(self, step):
        self.select_row(self.table.currentRow() + step)

    def show_current(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.filtered):
            self.index_label.setText("Index: -")
            self.position_label.setText(f"(0/{len(self.filtered)})")
            self.current_label.setText("")
            self.detail_text.clear()
            return
        i = self.filtered[row]
        item = self.data[i]
        self.index_label.setText(f"Index: {i}")
        selected = len(self.selected_rows())
        self.position_label.setText(
            f"({row + 1}/{len(self.filtered)}" + (f", {selected} selected)" if selected > 1 else ")"))
        self.current_label.setText(f"<b>{item.get('label') or 'unlabelled'}</b>")
        self.detail_text.setPlainText(item.get("text", ""))

    def selected_rows(self):
        return sorted(index.row() for index in self.table.selectionModel().selectedRows())

    def set_label(self, label):
        """Label the selected messages.

        One message: move on to the next one. Several: ask first, then keep them selected.
        Rows stay in the table until the next filter change, even if they no longer match.
        """
        rows = self.selected_rows()
        if not rows:
            return
        if len(rows) > 1 and QMessageBox.question(
                self, "Label selected", f"Label {len(rows)} selected messages as {label}?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        changed = 0
        for row in rows:
            i = self.filtered[row]
            if self.data[i].get("label") != label:
                self.data[i]["label"] = label
                self.table.item(row, 1).setText(label)
                changed += 1
        if changed:
            self.setWindowModified(True)

        if len(rows) > 1:
            self.statusBar().showMessage(f"{len(rows)} entries → {label} ({changed} changed)")
            self.show_current()
            return
        self.statusBar().showMessage(f"Entry {self.filtered[rows[0]]} → {label}")
        if rows[0] + 1 < len(self.filtered):
            self.select_row(rows[0] + 1)
        else:
            self.show_current()


def button(text, slot):
    widget = QPushButton(text)
    widget.clicked.connect(slot)
    return widget


def combo(values, slot):
    widget = QComboBox()
    widget.addItems(values)
    widget.currentIndexChanged.connect(slot)
    return widget


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SMSLabeler(*sys.argv[1:2])
    window.show()
    window.load_data(window.file_edit.text())
    sys.exit(app.exec())
