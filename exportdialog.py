from PyQt5.QtCore import pyqtSignal as Signal
from ui_exportdialog import Ui_ExportDialog
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QAbstractItemView, QPushButton, QDialogButtonBox, \
    QMessageBox
import numpy as np


class ExportDialog(QDialog, Ui_ExportDialog):
    start_export = Signal(list, list, str)

    def __init__(self, parent=None, n_frame=0, n_fov=0):
        super(ExportDialog, self).__init__()
        self.setParent(parent)
        self.setupUi(self)

        self.frames = n_frame
        self.fovs = n_fov

        self.fov_list = []
        self.frame_list = []

        self.file_path = None

        self.frame_range_label.setText(f"Frame range (0-{self.frames})")
        self.start_frame_line.setText("0")
        self.end_frame_line.setText(f"{self.frames}")
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.start_export_button.clicked.connect(self.start_export_data)
        self.fov_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.fov_list_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(n_fov):
            self.fov_list_widget.addItem(f"fov_{i}")
        self.select_all_box.clicked.connect(self.select_all)
        self.chose_path_button.clicked.connect(self.chose_path)

        self.close_button.clicked.connect(self.close_window)
        self.start_export.connect(self.close)

    def chose_path(self):
        self.file_path, _ = QFileDialog.getSaveFileName(self, "Save data", ".\\", "csv file (*.csv)")
        if self.file_path:
            self.save_path_line.setText(self.file_path + ".csv")
        else:
            self.file_path = None
            QMessageBox.warning(self, "Warning", "File path not valid.", QMessageBox.Ok, QMessageBox.Ok)

    def select_all(self):
        if self.select_all_box.isChecked():
            self.fov_list_widget.selectAll()
        else:
            self.fov_list_widget.clearSelection()

    def start_export_data(self):
        if self.file_path is None:
            return
        # collect all frame and fov names
        self.frame_list = []
        self.fov_list = []
        start_frame = int(self.start_frame_line.text())
        end_frame = int(self.end_frame_line.text())
        start_frame = np.clip(start_frame, 0, self.frames)
        end_frame = np.clip(end_frame, 0, self.frames + 1)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        self.frame_list = [i for i in range(start_frame, end_frame)]
        for i in self.fov_list_widget.selectedItems():
            n = int(i.text().split("_")[-1])
            self.fov_list.append(n)
        self.start_export.emit(self.fov_list, self.frame_list, self.file_path)

    def close_window(self):
        self.close()
