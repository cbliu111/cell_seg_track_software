import sys
import numpy as np
import h5py
from nd2reader import ND2Reader
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QAbstractItemView, QPushButton, QDialogButtonBox, QMessageBox
from PyQt5.QtCore import pyqtSignal as Signal
from ui_unet import Ui_Dialog
import skimage.exposure
from base import save_unet_seg_result, get_default_path
from yeaz.segment import segment
import yeaz.neural_network as nn
import yeaz.hungarian as hu


class UNetDialog(QDialog, Ui_Dialog):
    finished = Signal()

    def __init__(self, parent=None, frames=0, n_fov=0):
        super(UNetDialog, self).__init__()
        self.setParent(parent)
        self.setupUi(self)

        self.nd2files = None
        self.rows = 512
        self.cols = 512
        self.frames = frames
        self.fovs = n_fov
        self.num_frames_list = []
        self.fov_list = []
        self.frame_list = []
        self.hdfpath = None
        self.weight_path = None
        self.channel = None
        self.model = None
        self.threshold_value = None
        self.seed_distance = None

        self.frame_range_label.setText(f"Frame range (0-{frames})")
        self.start_frame_line.setText("0")
        self.end_frame_line.setText(f"{frames}")
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.start_run_button.clicked.connect(self.start_run)
        self.fov_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.fov_list_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(n_fov):
            self.fov_list_widget.addItem(f"fov_{i}")
        self.select_all_box.clicked.connect(self.select_all)
        self.chose_weight_button.clicked.connect(self.load_weight)

        self.close_button.clicked.connect(self.close_window)
        self.finished.connect(self.close)

    def get_image_2d(self, channel, fov, frame):
        sum_frames = 0
        for i, n in enumerate(self.num_frames_list):
            if frame < sum_frames + n:
                idx = frame - sum_frames
                with ND2Reader(self.nd2files[i]) as images:
                    images.default_coords["v"] = fov
                    images.default_coords["c"] = channel
                    return images[idx]
            else:
                sum_frames += n

    def load_weight(self):
        self.weight_path, _ = QFileDialog.getOpenFileName(self, "Open file", ".\\", "parameter file (*.pt, *.hdf5)")
        self.weight_path_line.setText(self.weight_path)

    def select_all(self):
        if self.select_all_box.isChecked():
            self.fov_list_widget.selectAll()
        else:
            self.fov_list_widget.clearSelection()

    def set_nd2files(self, nd2filelist, n_frames):
        self.nd2files = nd2filelist
        self.num_frames_list = n_frames
        images = ND2Reader(self.nd2files[0])
        self.rows = images.sizes["x"]
        self.cols = images.sizes["y"]

    def set_channel(self, c):
        self.channel = c

    def set_hdf_path(self, path):
        self.hdfpath = path

    def start_run(self):
        if not self.nd2files:
            QMessageBox.critical(self, "Warning", "No image data.", QMessageBox.Ok, QMessageBox.Ok)
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

        # get threshold and segment value
        self.threshold_value = float(self.threshold_line.text())
        self.seed_distance = int(self.seed_dist_line.text())

        # get label h5 file path
        if self.hdfpath is None:
            # use default path
            self.hdfpath = get_default_path(self.nd2files[0], ".h5")

        # get weight, return otherwise
        if self.weight_path is None:
            QMessageBox.critical(self, "Warning", "Please select weight file.", QMessageBox.Ok, QMessageBox.Ok)
            return

        # get channel, return if not
        if self.channel is None:
            QMessageBox.critical(self, "Warning", "No channel assigned.", QMessageBox.Ok, QMessageBox.Ok)
            return

        if not self.num_frames_list:
            QMessageBox.critical(self, "Warning", "Number of frames not assigned.", QMessageBox.Ok, QMessageBox.Ok)
            return

        # setup model
        self.model = nn.unet(pretrained_weights=self.weight_path, input_size=(None, None, 1))

        # starting loop
        steps = len(self.frame_list) * len(self.fov_list)
        progress = 0
        self.progressBar.setMaximum(steps)
        self.progressBar.show()
        for fov in self.fov_list:
            for frame in self.frame_list:
                image = self.get_image_2d(self.channel, fov, frame)
                seg_img = self.predict_threshold_segment(image)
                save_unet_seg_result(self.hdfpath, fov, frame, seg_img)
                progress += 1
                self.progressBar.setValue(progress)

        self.progressBar.hide()
        self.finished.emit()

    def predict_threshold_segment(self, image: np.ndarray):
        img = skimage.exposure.equalize_adapthist(image)
        img = img * 1.0

        # pad with zeros such that is divisible by 16
        rows, cols = img.shape
        row_add = 16 - rows % 16
        col_add = 16 - cols % 16
        padded_img = np.pad(img, ((0, row_add), (0, col_add)))

        # whole cell prediction with unet
        pred = self.model.predict(padded_img[np.newaxis, :, :, np.newaxis], batch_size=1)
        pred = pred[0, :, :, 0]
        pred = pred[:rows, :cols]

        # threshold
        if self.threshold_value is None:
            th_mask = nn.threshold(pred)
        else:
            th_mask = nn.threshold(pred, self.threshold_value)

        # segment
        seg = segment(th_mask, pred, self.seed_distance)
        return seg.astype(np.uint16)

    def close_window(self):
        self.finished.emit()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UNetDialog(frames=100, fovs=1000)
    win.show()
    sys.exit(app.exec())
