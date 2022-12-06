import os.path
import sys
import numpy as np
import pandas as pd
import skimage.morphology
from skimage.measure import regionprops
import h5py
from nd2reader import ND2Reader
import unet.hungarian as hu
from PyQt5.QtCore import QCoreApplication, QObject, QTimer, Qt, QThreadPool, pyqtSignal as Signal, pyqtSlot as Slot, \
    QPoint, QPointF, QDir
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QMenu, QHeaderView, QMessageBox, QStyle, \
    QAbstractItemView, QFileDialog, QPushButton, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QKeyEvent, QPen, QMouseEvent, QIcon, QPalette, QBrush
from base import get_label_color, label_statistics, save_label, get_default_path
from ui_labelwindow import Ui_LabelWindow
from importdialog import ImportDialog
from unetdialog import UNetDialog

sys.path.append("./unet")



class LabelWindow(QMainWindow, Ui_LabelWindow):
    def __init__(self):
        super(LabelWindow, self).__init__()
        self.setupUi(self)

        # data
        self.images = None
        # fov and time are stored as fov_i and frame_i, self.labels is the hdf file
        self.labels = None
        # save labels for undo and redo
        self.saved_labels = None
        # copied label for paste
        self.copied_label = None
        self.selected_label = []

        # status
        self.is_saved = True
        self.is_first_save = True
        # paths
        self.image_path = QDir.currentPath()
        self.nd2filepath = QDir.currentPath()
        self.hdfpath = QDir.currentPath()
        self.save_data_path = QDir.currentPath()

        self.num_frames = 0
        self.num_fov = 0
        self.num_channels = 0
        self.image_shape = (512, 512)
        self.fov = 0
        self.channel = 0
        # current frame index for label widget display
        self.frame_index = 0
        # store the max label value to avoid duplication of label value
        self.max_label_value = 0
        self.label_value = 0
        self.label_table_index = 0
        self.nd2_time_steps = None

        # menu
        self.actionOpen_nd2.triggered.connect(self.load_nd2)
        self.actionOpen_dir.triggered.connect(lambda: self.import_dialog.exec())
        self.actionLoad_hdf5.triggered.connect(self.load_other_hdf)
        self.actionSave.triggered.connect(self.save_current_label)
        self.actionExport_data.triggered.connect(self.export_data)
        self.actionExport_movie.triggered.connect(self.export_movie)
        self.actionExit.triggered.connect(self.close)
        self.actionLabel_opacity.triggered.connect(self.set_label_opacity)
        self.actionFill_holes.triggered.connect(self.fill_label_holes)
        self.actionRemove_small_pieces.triggered.connect(self.remove_label_small_pieces)
        self.actionArea_threshold.triggered.connect(self.threshold_label_area)
        self.actionUndo.triggered.connect(self.undo)
        self.actionRedo.triggered.connect(self.redo)
        self.actionShow_channel_overlay.triggered.connect(self.show_overlay)
        self.actionHide_label.triggered.connect(lambda: self.label_widget.hide_label())
        self.actionShow_label.triggered.connect(lambda: self.label_widget.show_label())
        self.actionShow_label_ID.triggered.connect(lambda: self.label_widget.show_label_id())
        self.actionHide_label_ID.triggered.connect(lambda: self.label_widget.hide_label_id())
        self.actionU_Net_segment.triggered.connect(self.segment_with_unet)
        self.actionRetrack_from_current.triggered.connect(self.retrack)

        # ROI group
        self.fov_box.currentIndexChanged.connect(self.set_fov)
        self.channel_box.currentIndexChanged.connect(self.set_channel)

        # tools group
        self.tool_box.setEnabled(True)
        self.tool_box.currentTextChanged.connect(lambda t: self.label_widget.set_brush_type(t))
        self.brush_size_box.setValue(50)
        self.brush_size_box.valueChanged.connect(self.set_brush_size)
        self.draw_button.clicked.connect(lambda: self.label_widget.set_draw())
        self.erase_button.clicked.connect(lambda: self.label_widget.set_erase())

        # change id group
        self.change_id_button.clicked.connect(self.change_label_id)
        self.exchange_id_button.clicked.connect(self.exchange_label_id)

        # copy and paste group
        self.copy_button.clicked.connect(self.copy_current_label)
        self.copy_all_button.clicked.connect(self.copy_all_label)
        self.paste_button.clicked.connect(self.paste)

        # label widget
        self.label_widget.read_next_frame.connect(self.jump_to_next_frame)
        self.label_widget.read_previous_frame.connect(self.jump_to_previous_frame)
        self.label_widget.label_updated.connect(lambda:
                                                self.generate_label_table_from_label(self.label_widget.render.label))
        self.label_widget.copy_id_at_mouse_position.connect(self.copy_label_at_mouse)
        self.label_widget.paste_id_inplace.connect(self.paste)
        self.label_widget.delete_id_at_mouse_position.connect(self.delete_label_at_mouse)
        self.label_widget.select_id_at_mouse_position.connect(self.select_label_at_mouse)
        self.pixmap_scale = 1
        self.brush_size = 50

        # label table
        self.label_table.clearContents()
        self.label_table.setStyleSheet("background-color:rgba(0,0,0,0)")
        # select whole row instead of cell
        self.label_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.label_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.label_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.label_table.resizeColumnsToContents()
        self.label_table.resizeRowsToContents()
        self.label_table.cellClicked.connect(self.label_table_select)
        self.new_label_button.clicked.connect(self.add_new_label)
        self.delete_label_button.clicked.connect(self.delete_label)

        # timer
        self.time_steps = [i for i in range(10)]
        self.timer = QTimer()
        self.timer.timeout.connect(self.jump_to_next_frame)

        # time label
        self.lineEditStartTime.setText("0")
        self.lineEditCurrentTime.setText("0")
        self.lineEditEndTime.setText(f"{self.time_steps[-1]}")
        self.lineEditTimeInterval.setText(f"{self.time_steps[1] - self.time_steps[0]}")

        # play buttons
        self.buttonPlay.clicked.connect(self.play)
        self.buttonFirstFrame.clicked.connect(lambda: self.jump_to_frame(0))
        self.buttonLastFrame.clicked.connect(lambda: self.jump_to_frame(self.num_frames - 1))
        self.buttonNextFrame.clicked.connect(self.jump_to_next_frame)
        self.buttonPreviousFrame.clicked.connect(self.jump_to_previous_frame)

        # view slider
        self.viewSlider.value_changed.connect(self.jump_to_frame)
        self.viewSlider.setRange(0, self.num_frames - 1)
        self.viewSlider.setValue(0)

        # dialog
        self.import_dialog = ImportDialog(None)
        self.import_dialog.data_loaded.connect(self.import_dir_data)

        self.show()

    def segment_with_unet(self):
        unet_dialog = UNetDialog(frames=self.num_frames, fovs=self.num_fov)
        unet_dialog.set_channel(self.channel)
        unet_dialog.set_images(self.images)
        unet_dialog.set_hdf_path(self.hdfpath)
        unet_dialog.finished.connect(self.update_label_display)
        unet_dialog.exec()

    def update_label_display(self):
        label = self.get_label_from_hdf()
        if label is None:
            label = np.zeros(self.image_shape, dtype=np.uint8)
        self.label_widget.set_label(label)
        self.generate_label_table_from_label(label)
        self.max_label_value = label.max()
        self.label_widget.show_label()
        if self.label_table.rowCount() > 1:
            self.label_table.selectRow(0)
            self.label_table_select()

    def track(self, fov, frame):
        file = h5py.File(self.hdfpath, "r+")
        if frame < 1:
            return
        if f"frame_{frame - 1}" in file[f"/fov_{fov}"]:
            # if there is a mask at previous frame
            prev = file[f"/fov_{fov}/frame_{frame - 1}"][:]
            # if there is a mask at current frame
            if f"frame_{frame}" in file[f"/fov_{fov}"]:
                curr = file[f"/fov_{fov}/frame_{frame}"][:]
                out = hu.correspondence(prev, curr)
            else:
                out = np.zeros_like(prev, dtype=np.uint8)
        else:
            if f"frame_{frame}" in file[f"/fov_{fov}"]:
                curr = file[f"/fov_{fov}/frame_{frame}"][:]
                out = curr
            else:
                out = np.zeros(self.image_shape, dtype=np.uint8)
        save_label(self.hdfpath, fov, frame, out.astype(np.uint8))

    def retrack(self):
        """retrack all frames from start to end"""
        for i in range(self.num_frames):
            self.track(self.fov, i)

    def fill_label_holes(self):
        """fill holes smaller than 64 pixels in each label area"""
        lb = self.label_widget.render.label
        for lv in np.unique(lb):
            if lv == 0:
                continue
            else:
                mask = lb == lv
                mask = skimage.morphology.remove_small_holes(mask, area_threshold=64)
                lb[mask] = lv
        self.label_widget.set_label(lb)
        self.label_widget.set_label_value(self.label_value)
        self.generate_label_table_from_label(lb)

    def remove_label_small_pieces(self):
        """remove pieces smaller than 64"""
        lb = self.label_widget.render.label
        mask = lb != 0
        mask = skimage.morphology.remove_small_objects(mask, min_size=64)
        lb = lb * mask
        self.label_widget.set_label(lb)
        self.label_widget.set_label_value(self.label_value)
        self.generate_label_table_from_label(lb)

    def threshold_label_area(self):
        lb = self.label_widget.render.label
        for lv in np.unique(lb):
            if lv == 0:
                continue
            else:
                mask = lb == lv
                s = mask.sum()
                if s > 4096 or s < 64:
                    lb[mask] = 0
        self.label_widget.set_label(lb)
        self.label_widget.set_label_value(self.label_value)
        self.generate_label_table_from_label(lb)

    def undo(self):
        # TODO:
        pass

    def redo(self):
        # TODO:
        pass

    def show_overlay(self):
        # TODO:
        pass

    def set_fov(self, f):
        self.fov = f
        self.update_default_coords()

    def set_label_opacity(self):
        # TODO: switch between transparent and solid
        pass

    def set_channel(self, c):
        self.channel = c
        self.update_default_coords()

    def set_brush_size(self, value):
        self.brush_size = value
        self.label_widget.set_brush_size(value)

    def update_default_coords(self):
        self.images.default_coords["v"] = self.fov
        self.images.default_coords["c"] = self.channel
        self.time_steps = self.nd2_time_steps[self.fov]
        image = self.images[self.frame_index]
        self.label_widget.set_image(image)
        self.update_label_display()

    def get_label_from_hdf(self):
        # fov and time are stored as fov group and frame dataset as /fov_i/frame_j
        file = h5py.File(self.hdfpath, "r")
        if f"frame_{self.frame_index}" in file[f"/fov_{self.fov}"]:
            label = file[f"/fov_{self.fov}/frame_{self.frame_index}"][:]
        else:
            label = None
        file.close()
        return label

    def create_hdf(self):
        file = h5py.File(self.hdfpath, "a")
        for i in range(self.num_fov):
            file.create_group(f"/fov_{i}")
        # create the first dataset
        d = np.zeros(self.image_shape, dtype=np.uint8)
        file.create_dataset(f"/fov_0/frame_0", data=d, compression="gzip")
        file.close()

    def save_current_label(self):
        lb = self.label_widget.render.label
        save_label(self.hdfpath, self.fov, self.frame_index, lb)

    def load_other_hdf(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open hdf5", ".\\", "hdf5 file (*.h5)")
        if file:
            self.hdfpath = file
            label = self.get_label_from_hdf()
            if label:
                self.label_widget.set_label(label)
            self.is_saved = True
            self.is_first_save = False
            QMessageBox.information(self, "Success", "hdf5 file loaded")
        else:
            QMessageBox.information(self, "Fail", "No file loaded")
            return

    def get_nd2_data(self):
        """get the file name of the nd2 file
        read all the necessary metadata
        create a hdf5 file if this is the first analysis
        load label from hdf5 file if file exist
        and update all related widgets
        hdf5 file is assumed to have the same file name with the nd2 file
        but with different postfix"""
        file, _ = QFileDialog.getOpenFileName(self, "Open nd2", ".\\", "nd2 file (*.nd2)")
        if file:
            # read nd2 file
            self.nd2filepath = file
            self.hdfpath = get_default_path(self.nd2filepath, ".h5")
            self.images = ND2Reader(self.nd2filepath)
            # obtain metadata
            self.num_fov = self.images.sizes["v"]
            self.num_frames = self.images.sizes["t"]
            self.num_channels = self.images.sizes["c"]
            self.image_shape = (self.images.sizes["x"], self.images.sizes["y"])
            # set current fov, channel and time step sequence
            # time steps are recorded sequentially for every capture
            self.fov = 0
            self.channel = 0
            self.frame_index = 0
            self.nd2_time_steps = self.images.timesteps.reshape(self.num_frames, self.num_fov).T
            self.update_default_coords()
            # if file exists, read, create otherwise
            exist = os.path.exists(self.hdfpath)
            if not exist:
                self.create_hdf()
            label = self.get_label_from_hdf()
            self.max_label_value = label.max()
            # regenerate label table
            self.generate_label_table_from_label(label)
            # update view slider
            self.viewSlider.setRange(0, self.num_frames - 1)
            self.viewSlider.setValue(0)
            # update combobox
            for c in self.images.metadata["channels"]:
                self.channel_box.addItem(c)
            for i in self.images.metadata["fields_of_view"]:
                self.fov_box.addItem(f"fov_{i}")
            # update times in unit of minutes
            self.lineEditStartTime.setText("0")
            t = self.time_steps[-1] / 60000
            self.lineEditEndTime.setText(f"{t : .2f}")
            t = t / self.num_frames
            self.lineEditTimeInterval.setText(f"{t : .2f}")
            self.lineEditCurrentTime.setText("0")
            # update label widgets
            self.label_widget.set_label(label)
            self.label_widget.show_label()
            self.label_widget.show_label_id()
            if self.label_table.rowCount() > 1:
                self.label_table.selectRow(0)
                self.label_table_select()
            self.is_saved = True
            self.is_first_save = False

    def load_nd2(self):
        if self.images or not self.is_saved:
            choice = QMessageBox.question(self, "Info", "Do you want to save current labels?",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if choice == QMessageBox.Yes:
                self.save_current_label()
                self.get_nd2_data()
                return
            elif choice == QMessageBox.No:
                self.get_nd2_data()
                return
            else:
                return
        else:
            self.get_nd2_data()
            return

    def export_data(self):
        label_list = []
        file, _ = QFileDialog.getSaveFileName(self, "Save csv", ".\\", "csv file (*.csv")
        if file:
            self.save_data_path = file
        else:
            self.save_data_path = get_default_path(self.nd2filepath, ".csv")
        for frame in range(self.num_frames):
            label = self.get_label_from_hdf()
            if label is None:
                continue
            regions = regionprops(label_image=label, intensity_image=self.images[frame])
            for prop in regions:
                stats = {
                    "Label": prop.label,
                    "Time": self.time_steps[frame],
                }
                prop.coords
            for value in np.unique(label):
                if value == 0:
                    continue
                else:
                    stats = {"Label": value,
                             "Time": self.time_steps[frame],
                             "Channel": self.channel,
                             **label_statistics(self.images[frame], label),
                             "Disappeared": not (value in self.selected_label)}
                    label_list.append(stats)
        df = pd.DataFrame(label_list)
        df = df.sort_values(["Label", "Time"])
        df.to_csv(self.save_data_path, index=False)

    def export_movie(self):
        # TODO: export overlay image as movie
        pass

    def import_dir_data(self, data, start_time, time_interval):
        self.images = data
        self.time_steps.append(start_time)
        for i in range(len(self.images)):
            self.time_steps.append(i * time_interval)
        if self.images is None:
            QMessageBox.critical(self, "Error", "Data is empty")
            return
        else:
            self.is_saved = False
            self.is_first_save = True
            self.image_path = QDir.currentPath()
            # set slider range to the number of images of first channel
            num_images = len(self.images[0])
            self.viewSlider.setRange(0, num_images - 1)
            self.lineEditStartTime.setText(f"{self.time_steps[0]}")
            self.lineEditTimeInterval.setText(f"{time_interval}")
            self.lineEditCurrentTime.setText(f"{self.time_steps[0]}")
            self.lineEditEndTime.setText(f"{self.time_steps[-1]}")

    def play(self):
        if self.images is None:
            return
        elif self.buttonPlay.text() == "Play":
            self.buttonPlay.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaStop))
            self.buttonPlay.setText("Stop")
            self.timer.start(200)
        else:
            self.timer.stop()
            self.buttonPlay.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))

    def select_label_at_mouse(self, x, y):
        label = self.label_widget.render.label
        lv = label[x, y]
        items = self.label_table.findItems(f"{lv}", Qt.MatchExactly)
        if len(items) > 0:
            item = items[0]
            row = item.row()
            self.label_table_index = row
            c = get_label_color(lv)
            self.label_table.setStyleSheet(f"selection-background-color: {c}")
            self.label_table.selectRow(row)
            self.label_table.verticalScrollBar().setSliderPosition(row)

    def copy_label_at_mouse(self, x, y):
        label = self.label_widget.render.label
        lv = label[x, y]
        self.copied_label = np.where(label == lv, lv, 0)

    def delete_label_at_mouse(self, x, y):
        label = self.label_widget.render.label
        lv = label[x, y]
        if lv == 0:
            return
        # remove label id in label
        lb = self.label_widget.render.label
        lb[lb == lv] = 0
        self.label_widget.set_label(lb)
        # find index of lv in label table
        # remove row corresponds to lv
        items = self.label_table.findItems(f"{lv}", Qt.MatchExactly)
        if len(items) > 0:
            item = items[0]
            row = item.row()
            self.label_table.removeRow(row)

    def jump_to_next_frame(self):
        if self.frame_index == self.num_frames - 1:
            self.jump_to_frame(0)
        else:
            self.jump_to_frame(self.frame_index + 1)

    def jump_to_previous_frame(self):
        if self.frame_index == 0:
            self.jump_to_frame(self.num_frames - 1)
        else:
            self.jump_to_frame(self.frame_index - 1)

    def jump_to_frame(self, index: int):
        if self.images is None:
            return
        elif 0 <= index < self.num_frames:
            # save label when jump to another frame
            self.save_current_label()
            # jump and update widgets
            self.frame_index = index
            self.pixmap_scale = self.label_widget.scale
            self.brush_size = self.label_widget.brush_size
            self.viewSlider.setValue(index)
            self.label_widget.set_image(self.images[index])
            label = self.get_label_from_hdf()
            if label is None:
                label = np.zeros(self.image_shape, dtype=np.uint8)
            self.max_label_value = label.max()
            self.label_widget.set_label(label)
            self.generate_label_table_from_label(label)
            self.label_widget.set_scale(self.pixmap_scale)
            self.label_widget.set_brush_size(self.brush_size)
            self.label_widget.set_label_value(self.label_value)
            self.label_widget.show_image()
            t = self.time_steps[index] / 60000
            self.lineEditCurrentTime.setText(f"{t : .2f}")
            self.viewSlider.setValue(index)

    def label_table_select(self):
        row = self.label_table.currentRow()
        self.label_table_index = self.label_table.currentRow()
        item = self.label_table.item(row, 0)
        self.label_value = int(item.text())
        self.label_widget.set_label_value(self.label_value)
        c = get_label_color(self.label_value)
        self.label_table.setStyleSheet(f"selection-background-color: {c}")
        self.label_table.selectRow(row)
        self.label_table.update()

    def add_new_label(self):
        self.label_value = self.max_label_value + 1
        self.max_label_value = self.label_value
        self.label_widget.set_label_value(self.label_value)
        self.label_widget.show_label()
        rows = self.label_table.rowCount()
        self.label_table.insertRow(rows)
        v = QTableWidgetItem(f"{self.label_value}")
        c = get_label_color(self.label_value)
        b = QTableWidgetItem("")
        b.setBackground(QColor(c))
        self.label_table.setItem(rows, 0, v)
        self.label_table.setItem(rows, 1, b)
        self.label_table.setStyleSheet(f"selection-background-color: {c}")
        self.label_table.selectRow(rows)

    def generate_label_table_from_label(self, label):
        self.label_table.clearContents()
        self.label_table.setRowCount(0)
        for i, lv in enumerate(np.unique(label)):
            if lv == 0:
                continue
            self.label_table.setRowCount(i)
            v = QTableWidgetItem(f"{lv}")
            c = get_label_color(lv)
            b = QTableWidgetItem("")
            b.setBackground(QColor(c))
            self.label_table.setItem(i - 1, 0, v)
            self.label_table.setItem(i - 1, 1, b)
            self.label_table.setStyleSheet(f"selection-background-color: {c}")

    def change_label_id(self):
        if self.first_id_edit.text() and self.second_id_edit.text():
            id1 = int(self.first_id_edit.text())
            id2 = int(self.second_id_edit.text())
            self.max_label_value = id2 if id2 > self.max_label_value else self.max_label_value
            label = self.label_widget.render.label
            label[label == id1] = id2
            self.generate_label_table_from_label(label)
            self.label_widget.set_label(label)

    def exchange_label_id(self):
        if self.first_id_edit.text() and self.second_id_edit.text():
            id1 = int(self.first_id_edit.text())
            id2 = int(self.second_id_edit.text())
            label = self.label_widget.render.label
            lvs = np.unique(label)
            if id1 not in lvs or id2 not in lvs:
                return
            label_copy = label.copy()
            label[label_copy == id1] = id2
            label[label_copy == id2] = id1
            self.generate_label_table_from_label(label)
            self.label_widget.set_label(label)

    def copy_current_label(self):
        label = self.label_widget.render.label
        self.copied_label = np.where(label == self.label_value, self.label_value, 0)

    def copy_all_label(self):
        self.copied_label = self.label_widget.render.label

    def paste(self):
        label = self.label_widget.render.label
        mask = self.copied_label > 0
        label[mask] = self.copied_label[mask]
        self.label_widget.set_label(label)
        lv = self.copied_label.max()
        if lv > self.max_label_value:
            self.max_label_value = lv
        # if already has the same label value in label table do nothing
        # otherwise add a new row
        items = self.label_table.findItems(f"{lv}", Qt.MatchExactly)
        if len(items) > 0:
            return
        # add a new row
        rows = self.label_table.rowCount()
        self.label_table.insertRow(rows)
        v = QTableWidgetItem(f"{lv}")
        c = get_label_color(lv)
        b = QTableWidgetItem("")
        b.setBackground(QColor(c))
        self.label_table.setItem(rows, 0, v)
        self.label_table.setItem(rows, 1, b)
        self.label_table.setStyleSheet(f"selection-background-color: {c}")
        self.label_table.selectRow(rows)

    def delete_label(self):
        if self.label_table.rowCount() == 0:
            return
        cr = self.label_table.currentRow()
        lv = int(self.label_table.item(cr, 0).text())
        self.label_table.removeRow(cr)
        lb = self.label_widget.render.label
        lb[lb == lv] = 0
        self.label_widget.set_label(lb)

    def closeEvent(self, event):
        if not self.is_saved:
            choice = QMessageBox.question(self, "Info", "Confirm to close?", QMessageBox.Yes | QMessageBox.Cancel)
            if choice == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()


if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = LabelWindow()
    sys.exit(app.exec())
