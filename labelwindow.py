import os.path
import sys
import numpy as np
import pandas as pd
import skimage.morphology
from skimage.measure import regionprops
import h5py
from nd2reader import ND2Reader
from yeaz import hungarian as hu
from PyQt5.QtCore import QCoreApplication, QObject, QTimer, Qt, QThreadPool, pyqtSignal as Signal, pyqtSlot as Slot, \
    QPoint, QPointF, QDir
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QMenu, QHeaderView, QMessageBox, QStyle, \
    QAbstractItemView, QFileDialog, QPushButton, QTableWidgetItem, QProgressBar
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QKeyEvent, QPen, QMouseEvent, QIcon, QPalette, QBrush
from base import get_label_color, save_label, get_default_path, get_label_from_hdf
from ui_labelwindow import Ui_LabelWindow
from importdialog import ImportDialog
from unetdialog import UNetDialog

sys.path.append("./unet")


class LabelWindow(QMainWindow, Ui_LabelWindow):
    """
    The main label window for cell segmentation, tracking and data extracting.
    """
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
        self.actionRetrack_from_current.triggered.connect(self.retrack_from_current)
        self.actionRetrack_from_first.triggered.connect(self.retrack_from_first)

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
        self.label_widget.draw_at_mouse_position.connect(self.draw_at_mouse)
        self.label_widget.send_zoom_point.connect(self.zoom_view_widget)
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

        # track
        self.new_cell_id_for_track = 0

        # progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.statusbar.addWidget(self.progress_bar)

    def segment_with_unet(self):
        unet_dialog = UNetDialog(frames=self.num_frames, fovs=self.num_fov)
        unet_dialog.set_channel(self.channel)
        unet_dialog.set_images(self.images)
        unet_dialog.set_hdf_path(self.hdfpath)
        unet_dialog.finished.connect(self.update_label_display)
        unet_dialog.exec()

    def update_label_display(self):
        label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
        if label is None:
            label = np.zeros(self.image_shape, dtype=np.uint16)
        self.label_widget.set_label(label)
        self.generate_label_table_from_label(label)
        self.max_label_value = np.amax(label)
        self.label_widget.show_label()
        if self.label_table.rowCount() > 1:
            self.label_table.selectRow(0)
            self.label_table_select()

    def track(self, fov, frame):
        file = h5py.File(self.hdfpath, "r+")
        if frame < 1:
            return
        # if there is a label at previous frame
        if f"frame_{frame - 1}" in file[f"/fov_{fov}"]:
            prev = file[f"/fov_{fov}/frame_{frame - 1}"][:]
            # if there is a label at current frame, do track
            if f"frame_{frame}" in file[f"/fov_{fov}"]:
                curr = file[f"/fov_{fov}/frame_{frame}"][:]
                out = hu.correspondence(prev, curr, self.new_cell_id_for_track)
                # New cells should be given the unique identifier starting at the maximum label
                # value of all historical frames.
                if np.amax(out) >= self.new_cell_id_for_track:
                    self.new_cell_id_for_track = np.amax(out) + 1
            # if currently no label, return empty label
            else:
                out = np.zeros_like(prev, dtype=np.uint16)
        else:
            # if no label at previous frame, but has a label at current, return current label
            if f"frame_{frame}" in file[f"/fov_{fov}"]:
                curr = file[f"/fov_{fov}/frame_{frame}"][:]
                out = curr
            # if no label at previous and current frame, return empty label
            else:
                out = np.zeros(self.image_shape, dtype=np.uint16)
        save_label(self.hdfpath, fov, frame, out.astype(np.uint16))

    def fill_label_holes(self):
        """
        Fill holes smaller than 64 pixels in each label area.
        """
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
        """
        Remove pieces smaller than 64.
        """
        lb = self.label_widget.render.label
        mask = lb != 0
        mask = skimage.morphology.remove_small_objects(mask, min_size=64)
        lb = lb * mask
        self.label_widget.set_label(lb)
        self.label_widget.set_label_value(self.label_value)
        self.generate_label_table_from_label(lb)

    def threshold_label_area(self):
        """
        Remove label with size larger than 4096 or smaller than 64 for all the frames.
        """
        file = h5py.File(self.hdfpath, "r+")
        for frame in range(self.num_frames):
            if f"frame_{frame}" in file[f"/fov_{self.fov}"]:
                label = file[f"/fov_{self.fov}/frame_{frame}"][:]
            else:
                label = None
            if label is None:
                continue
            for lv in np.unique(label):
                if lv == 0:
                    continue
                else:
                    mask = label == lv
                    s = mask.sum()
                    if s > 4096 or s < 64:
                        label[mask] = 0
            if f"frame_{frame}" in file[f"/fov_{self.fov}"]:
                dataset = file[f"/fov_{self.fov}/frame_{frame}"]
                dataset[:] = label.astype(np.uint16)
            else:
                file.create_dataset(f"/fov_{self.fov}/frame_{frame}", data=label, compression="gzip")
        file.close()
        label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
        self.label_widget.set_label(label)
        self.set_view_labels()
        self.label_widget.set_label_value(self.label_value)
        self.generate_label_table_from_label(label)

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
        """
        Update default coords (v, c) of the images read from nd2 file.
        Update time steps, label widget and label table accordingly.
        """
        self.images.default_coords["v"] = self.fov
        self.images.default_coords["c"] = self.channel
        self.time_steps = self.nd2_time_steps[self.fov]
        image = self.images[self.frame_index]
        self.label_widget.set_image(image)
        self.set_view_images()
        self.update_label_display()

    def create_hdf(self):
        """
        When an hdf is not exists,
        create it at the hdfpath.
        Generate all groups corresponding to fovs,
        also generate the first label at frame = 0 for current fov.
        """
        file = h5py.File(self.hdfpath, "a")
        for i in range(self.num_fov):
            file.create_group(f"/fov_{i}")
        # create the first dataset
        d = np.zeros(self.image_shape, dtype=np.uint16)
        file.create_dataset(f"/fov_0/frame_0", data=d, compression="gzip")
        file.close()

    def save_current_label(self):
        lb = self.label_widget.render.label
        save_label(self.hdfpath, self.fov, self.frame_index, lb)

    def load_other_hdf(self):
        """
        Load other hdf file instead of the default hdfpath.
        """
        file, _ = QFileDialog.getOpenFileName(self, "Open hdf5", ".\\", "hdf5 file (*.h5)")
        if file:
            self.hdfpath = file
            label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
            if label:
                self.label_widget.set_label(label)
                self.set_view_labels()
            self.is_saved = True
            self.is_first_save = False
            QMessageBox.information(self, "Success", "hdf5 file loaded")
        else:
            QMessageBox.information(self, "Fail", "No file loaded")
            return

    def get_nd2_data(self):
        """
        Get the file name of the nd2 file.
        Read all the necessary metadata.
        Create a hdf5 file if this is the first analysis.
        Load label from hdf5 file if file exist.
        and update all related widgets.
        hdf5 file is assumed to have the same file name with the nd2 file,
        but with different postfix.
        """
        file, _ = QFileDialog.getOpenFileName(self, "Open nd2", ".\\", "nd2 file (*.nd2)")
        if file:
            # read nd2 file
            self.nd2filepath = file
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
            # if file exists, read, create otherwise
            self.hdfpath = get_default_path(self.nd2filepath, ".h5")
            exist = os.path.exists(self.hdfpath)
            if not exist:
                self.create_hdf()
            # update default coords
            self.update_default_coords()
            # update max label value
            label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
            self.max_label_value = np.amax(label)
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
            self.set_view_labels()
            self.label_widget.show_label()
            self.view_widget_left.show_label()
            self.view_widget_right.show_label()
            self.label_widget.show_label_id()
            self.view_widget_left.show_label_id()
            self.view_widget_right.show_label_id()
            if self.label_table.rowCount() > 1:
                self.label_table.selectRow(0)
                self.label_table_select()
            self.jump_to_frame(0)
            self.is_saved = True
            self.is_first_save = False

    def load_nd2(self):
        if self.images or not self.is_saved:
            choice = QMessageBox.question(self, "Info", "Do you want to save current labels?",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if choice == QMessageBox.Yes:
                self.save_current_label()
                self.channel_box.clear()
                self.fov_box.clear()
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

    def retrack_from_first(self):
        """
        Retrack all frames from first frame to end.
        If current frame does not have a label, do nothing.
        The first frame label is relabeled to remove missing cells caused by 
        manual corrections.
        """
        # locate the first frame that has a label
        file = h5py.File(self.hdfpath, "r")
        start_frame = -1
        for i in range(self.num_frames):
            if f"frame_{i}" in file[f"/fov_{self.fov}"]:
                start_frame = i
                break
        if start_frame == -1:
            return
        else:
            label = file[f"/fov_{self.fov}/frame_{start_frame}"][:]
            # rearrange all the labels starting by 1
            new_lv = 1
            label_copy = label.copy()
            for lv in np.unique(label):
                if lv == 0:
                    continue
                label[label_copy == lv] = new_lv
                new_lv += 1
        file.close()
        self.new_cell_id_for_track = np.amax(label) + 1
        # jump to the first frame with label and refresh label
        self.jump_to_frame(start_frame)
        self.label_widget.set_label(label)
        self.set_view_labels()
        self.generate_label_table_from_label(label)
        # save the first frame that has a label, and start tracking
        save_label(self.hdfpath, self.fov, start_frame, label.astype(np.uint16))
        self.progress_bar.setRange(start_frame, self.num_frames)
        self.progress_bar.show()
        for i in range(start_frame, self.num_frames):
            self.progress_bar.setValue(i)
            self.track(self.fov, i)
        self.progress_bar.hide()

    def retrack_from_current(self):
        """
        Retrack all frames from current to end.
        If current frame does not have a label, do nothing
        """
        # locate the first frame that has a label
        self.progress_bar.setRange(self.frame_index, self.num_frames)
        self.progress_bar.show()
        label = self.label_widget.render.label
        self.new_cell_id_for_track = np.amax(label) + 1
        for i in range(self.frame_index, self.num_frames):
            self.progress_bar.setValue(i)
            self.track(self.fov, i)
        label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
        self.label_widget.set_label(label)
        self.set_view_labels()
        self.generate_label_table_from_label(label)
        self.progress_bar.hide()

    def export_data(self):
        """
        Export label statistics of current fov.
        Different fovs may be exported to different files.
        """
        label_list = []
        file, _ = QFileDialog.getSaveFileName(self, "Save csv", ".\\", "csv file (*.csv)")
        if file:
            self.save_data_path = file
        else:
            self.save_data_path = get_default_path(self.nd2filepath, ".csv")

        file = h5py.File(self.hdfpath, "r")

        self.progress_bar.setRange(0, self.num_frames)
        for frame in range(self.num_frames):
            self.progress_bar.setValue(frame)
            if f"frame_{frame}" in file[f"/fov_{self.fov}"]:
                label = file[f"/fov_{self.fov}/frame_{frame}"][:]
            else:
                label = None
            if label is None or np.amax(label) == 0:
                continue
            regions = regionprops(label_image=label)
            for prop in regions:
                x, y = prop.centroid
                stats = {
                    "Label": prop.label,
                    "Frame": frame,
                    "Time": self.time_steps[frame],
                    "Area": prop.area,
                    "Centroid X": x,
                    "Centroid y": y,
                    "Orientation": prop.orientation,
                    "Length major": prop.axis_major_length,
                    "Length minor": prop.axis_minor_length,
                }
                for i, name in enumerate(self.images.metadata["channels"]):
                    fl_img = self.images.get_frame_2D(c=i, t=frame, v=self.fov)
                    ix = prop.coords[:, 0]
                    iy = prop.coords[:, 1]
                    fl_values = fl_img[ix, iy]
                    stats[f"Mean intensity of {name}"] = fl_values.mean()
                    stats[f"Total intensity of {name}"] = fl_values.sum()
                    stats[f"Intensity variance of {name}"] = fl_values.var()
                label_list.append(stats)

        file.close()
        df = pd.DataFrame(label_list)
        df = df.sort_values(["Label", "Time"])
        df.to_csv(self.save_data_path, index=False)

    def export_movie(self):
        # TODO: export overlay image as movie, use skimage gray2rgb
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
            self.buttonPlay.setText("Stop")
            self.buttonPlay.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaStop))
            self.timer.start(200)
        else:
            self.buttonPlay.setText("Play")
            self.buttonPlay.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()

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

    def zoom_view_widget(self, p: QPointF):
        p = p / 600 * 512
        self.view_widget_left.zoom_point = p
        self.view_widget_right.zoom_point = p
        self.view_widget_left.zoom(self.label_widget.scale)
        self.view_widget_right.zoom(self.label_widget.scale)

    def draw_at_mouse(self, x, y):
        """
        If label value at mouse is not 0, 
        change to the value and color of the label value.
        Otherwise a new label is generated."""
        label = self.label_widget.render.label
        lv = label[x, y]
        # if draw at empty, create a new label
        if lv == 0:
            self.add_new_label()
        else:
            self.label_widget.set_label_value(lv)

    def copy_label_at_mouse(self, x, y):
        label = self.label_widget.render.label
        lv = label[x, y]
        self.copied_label = np.where(label == lv, lv, 0)

    def delete_label_at_mouse(self, x, y):
        """
        Delete all pixels having the label value at mouse holding.
        """
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

    def set_view_images(self):
        """
        Set all images for left view widget and right view widget.
        Index bound is checked for view widget
        """
        default_img = np.zeros(self.image_shape, dtype=np.uint16)
        if self.frame_index > 0:
            self.view_widget_left.set_image(self.images[self.frame_index - 1])
        else:
            self.view_widget_left.set_image(default_img)
        if self.frame_index < self.num_frames - 1:
            self.view_widget_right.set_image(self.images[self.frame_index + 1])
        else:
            self.view_widget_right.set_image(default_img)

    def set_view_labels(self):
        default_lb = np.zeros(self.image_shape, dtype=np.uint16)
        llb = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index - 1)
        if llb is None or self.frame_index == 0:
            self.view_widget_left.set_label(default_lb)
        else:
            self.view_widget_left.set_label(llb)
        rlb = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index + 1)
        if rlb is None or self.frame_index == self.num_frames - 1:
            self.view_widget_right.set_label(default_lb)
        else:
            self.view_widget_right.set_label(rlb)

    def jump_to_frame(self, index: int):
        """
        Jump to frame, and update widgets accordingly. 
        """
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
            self.set_view_images()
            label = get_label_from_hdf(self.hdfpath, self.fov, self.frame_index)
            if label is None:
                label = np.zeros(self.image_shape, dtype=np.uint16)
            self.max_label_value = np.amax(label)
            self.label_widget.set_label(label)
            self.set_view_labels()
            self.generate_label_table_from_label(label)
            self.label_widget.set_scale(self.pixmap_scale)
            self.label_widget.set_brush_size(self.brush_size)
            self.label_widget.set_erase()
            self.label_widget.show_image()
            t = self.time_steps[index] / 60000
            self.lineEditCurrentTime.setText(f"{t : .2f}")
            self.viewSlider.setValue(index)

    def label_table_select(self):
        """
        Select label table and make sure the color of selected item is not 
        covered by selection color.
        """
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
        """
        New label is added to be maximum label value + 1.
        For missing label can be corresponding to cell that flushed out.
        """
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
        """
        Update label table completely from label.
        Label table is just a representation of the label stored in h5 file.
        """
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
        """
        After paste, should update the maximum label value,
        and also generate a new label table row if pasted label is 
        not exists in current frame.
        """
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
        """
        Delete label at the position mouse hanging.
        """
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
