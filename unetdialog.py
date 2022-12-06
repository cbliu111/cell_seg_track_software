import sys
import numpy as np
import h5py
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QAbstractItemView, QPushButton, QDialogButtonBox
from PyQt5.QtCore import pyqtSignal as Signal
from ui_unet import Ui_Dialog
import skimage.exposure
from base import save_label, get_default_path
from unet.segment import segment
import unet.neural_network as nn
import unet.hungarian as hu


class UNetDialog(QDialog, Ui_Dialog):
    finished = Signal()

    def __init__(self, parent=None, frames=0, fovs=0):
        super(UNetDialog, self).__init__()
        self.setParent(parent)
        self.setupUi(self)

        self.images = None
        self.rows = 512
        self.cols = 512
        self.frames = frames
        self.fovs = fovs
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
        self.end_frame_line.setText(f"{frames - 1}")
        self.progressBar.setValue(0)
        self.start_run_button.clicked.connect(self.start_run)
        self.fov_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.fov_list_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(fovs):
            self.fov_list_widget.addItem(f"fov_{i}")
        self.select_all_box.clicked.connect(self.select_all)
        self.chose_weight_button.clicked.connect(self.load_weight)

        self.close_button.clicked.connect(self.close_window)

    def load_weight(self):
        self.weight_path, _ = QFileDialog.getOpenFileName(self, "Open file", ".\\", "parameter file (*.pt, *.hdf5)")
        self.weight_path_line.setText(self.weight_path)

    def select_all(self):
        if self.select_all_box.isChecked():
            self.fov_list_widget.selectAll()
        else:
            self.fov_list_widget.clearSelection()

    def set_images(self, imgs):
        self.images = imgs
        self.rows, self.cols = self.images.frame_shape

    def set_channel(self, c):
        self.channel = c

    def set_hdf_path(self, path):
        self.hdfpath = path

    def start_run(self):
        # collect all frame and fov names
        self.frame_list = []
        self.fov_list = []
        start_frame = int(self.start_frame_line.text())
        end_frame = int(self.end_frame_line.text())
        start_frame = np.clip(start_frame, 0, self.frames)
        end_frame = np.clip(end_frame, 0, self.frames)
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
            self.hdfpath = get_default_path(self.images.filename, ".h5")

        # get weight, return otherwise
        if self.weight_path is None:
            return

        # get channel, return if not
        if self.channel is None:
            return
        else:
            self.images.default_coords["c"] = self.channel

        # setup model
        self.model = nn.unet(pretrained_weights=self.weight_path, input_size=(None, None, 1))

        # starting loop
        steps = len(self.frame_list) * len(self.fov_list)
        progress = 0
        self.progressBar.setMaximum(steps)
        for fov in self.fov_list:
            self.images.default_coords["v"] = fov
            for frame in self.frame_list:
                image = self.images[frame]
                seg_img = self.predict_threshold_segment(image)
                save_label(self.hdfpath, fov, frame, seg_img.astype(np.uint8))
                self.track(fov, frame)
                progress += 1
                self.progressBar.setValue(progress)

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
        pred = pred[0, :rows, :cols, 0]

        # threshold
        if self.threshold_value is None:
            th_mask = nn.threshold(pred)
        else:
            th_mask = nn.threshold(pred, self.threshold_value)

        # segment
        seg = segment(th_mask, pred, self.seed_distance)
        return seg.astype(np.uint8)

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
                out = np.zeros_like(prev)
        else:
            if f"frame_{frame}" in file[f"/fov_{fov}"]:
                curr = file[f"/fov_{fov}/frame_{frame}"][:]
                out = curr
            else:
                out = np.zeros((self.rows, self.cols))
        save_label(self.hdfpath, fov, frame, out.astype(np.uint8))

    def close_window(self):
        self.finished.emit()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UNetDialog(frames=100, fovs=1000)
    win.show()
    sys.exit(app.exec())
