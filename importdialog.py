import sys
import numpy as np
from skimage import io
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QAbstractItemView, QTableWidgetItem
from PyQt5.QtCore import QDir, Qt, pyqtSignal as Signal
from ui_importdialog import Ui_ImportDialog


class ImportDialog(QDialog, Ui_ImportDialog):
    """
    Dialog for import images from a directory.
    Use nd2 file loading whenever possible. 
    Precise time step information and other metadata are contained in the nd2 file.
    """
    data_loaded = Signal(list, list, float, float)  # images, colors, start_time, time_interval

    def __init__(self, parent=None):
        super(ImportDialog, self).__init__()
        self.setParent(parent)
        self.setupUi(self)
        self.setWindowTitle("Import Dir")
        self.images = []
        self.start_time = 0
        self.time_interval = 0
        self.names = []
        self.paths = []
        self.colors = []
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.tableWidget.clearContents()
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.saved_path = QDir.currentPath()

        # signals slots
        self.buttonAddChannel.clicked.connect(self.add_channel)
        self.buttonEditChannelPath.clicked.connect(self.edit_channel)
        self.buttonDeleteChannel.clicked.connect(self.delete_channel)
        self.buttonOK.clicked.connect(self.confirm)
        self.buttonCancel.clicked.connect(self.cancel)
        self.data_loaded.connect(self.close)

    def set_names_paths(self, names, paths):
        n = len(names)
        self.tableWidget.setRowCount(n)
        for i in range(n):
            name = QTableWidgetItem(names[i])
            path = QTableWidgetItem(paths[i])
            self.tableWidget.setItem(i, 0, name)
            self.tableWidget.setItem(i, 1, path)

    def set_time_interval(self, t):
        self.lineEditTimeInterval.setText(f"{t}")

    def set_start_time(self, t):
        self.lineEditStartTime.setText(f"{t}")

    def add_channel(self):
        current_path = QDir()
        if self.saved_path:
            path = QDir(self.saved_path)
            path.cdUp()
            p = QFileDialog.getExistingDirectory(self, "Load data", path.path())
            current_path.setPath(p)
            self.saved_path = current_path.path()
        else:
            p = QFileDialog.getExistingDirectory(self, "Load data", QDir.currentPath())
            current_path.setPath(p)
            self.saved_path = current_path.path()
        rows = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(rows + 1)
        path_item = QTableWidgetItem(current_path.path())
        self.tableWidget.setItem(rows, 1, path_item)

    def edit_channel(self):
        current_row = self.tableWidget.currentRow()
        current_path = QDir()
        if self.saved_path:
            path = QDir(self.saved_path)
            path.cdUp()
            current_path.setPath(QFileDialog.getExistingDirectory(self, "Load data", path.path()))
            if current_path.path():
                self.saved_path = current_path.path()
        else:
            current_path.setPath(QFileDialog.getExistingDirectory(self, "Load data", QDir.currentPath()))
            if current_path.path():
                self.saved_path = current_path.path()
        path_item = QTableWidgetItem(current_path.path())
        self.tableWidget.setItem(current_row, 1, path_item)

    def delete_channel(self):
        current_row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(current_row)

    def confirm(self):
        self.names.clear()
        self.paths.clear()
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            if self.tableWidget.item(i, 0) is None:
                QMessageBox.critical(self, "Error", f"Empty name in row {i}")
                return
            if self.tableWidget.item(i, 1) is None:
                QMessageBox.critical(self, "Error", f"Empty path in row {i}")
                return
            if self.tableWidget.item(i, 2) is None:
                QMessageBox.critical(self, "Error", f"Empty color in row {i}")
                return
            name = self.tableWidget.item(i, 0).text()
            path = self.tableWidget.item(i, 1).text()
            color = self.tableWidget.item(i, 2).text()
            self.names.append(name)
            self.paths.append(path)
            self.colors.append(color)
        if len(self.names) == 0 or len(self.paths) == 0:
            QMessageBox.critical(self, "Error", "No data selected")
            return
        if len(set(self.names)) != len(self.names):
            QMessageBox.critical(self, "Error", "Duplicated channel names")
            return
        if len(set(self.paths)) != len(self.paths):
            QMessageBox.critical(self, "Error", "Duplicated channel paths")
            return
        if len(self.names) != len(self.paths):
            QMessageBox.critical(self, "Error", "Number of names is not equal to number of paths")
            return
        # load images
        name_filters = ["*.tif"]
        num_file = 0
        for i, path in enumerate(self.paths):
            files = path.entryList(name_filters, QDir.Files | QDir.Readable, QDir.Name)
            if i == 0:
                num_file = len(files)
            elif len(files) != num_file:
                QMessageBox.critical(self, "Error", "Image number are not equal in each directory. Abort")
                self.close()
                return
            imgs = []
            for file in files:
                img = io.imread(path + "/" + file).astype(np.uint16)
                imgs.append(img)
            self.images.append(imgs)
        if self.lineEditStartTime.text():
            self.start_time = float(self.lineEditStartTime.text())
        if self.lineEditTimeInterval.text():
            self.time_interval = float(self.lineEditTimeInterval.text())
        QMessageBox.information(self, "Success", "Data loaded")
        self.data_loaded.emit(self.images, self.start_time, self.time_interval)

    def cancel(self):
        self.close()
