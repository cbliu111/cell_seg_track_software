import sys
import pandas as pd
import math
from skimage.measure import regionprops
import skimage.color
import numpy as np
import h5py
from PyQt5.QtGui import qRgb, QColor, QImage

DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                  'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')


def get_label_color(lv):
    """
    Return the default colors for label values.
    If label value is larger than the number of colors, 
    first color is reused. 
    Label value starts at 1, 0 is treated as background.  
    """
    return DEFAULT_COLORS[(lv - 1) % len(DEFAULT_COLORS)]


COLOR_TABLE = [QColor(i).rgb() for i in DEFAULT_COLORS] * 30
COLOR_TABLE = [qRgb(0, 0, 0)] + COLOR_TABLE


def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    color = skimage.color.gray2rgb(image)
    hsv = skimage.color.rgb2hsv(color)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return skimage.color.hsv2rgb(hsv)


def numpy_to_image(m: np.ndarray, fmt: QImage.Format):
    """
    Convert a np.ndarray to QImage with QImage format fmt.
    QImage is returned.
    If m dtype is np.uint16 for label, but fmt is QImage.Format_Grayscale8.
    If m dtype is np.uint16 for 16-bit image, fmt is QImage.Format_Grayscale16.
    """
    return QImage(m.data, m.shape[0], m.shape[1], m.strides[0], fmt)


def get_label_centers(label: np.ndarray):
    """
    Return labels, x, y coordinates of centroid
    by using skimage regionprops.
    Used for draw the label values on LabelWidget pixmap.
    """
    vals = []
    cx = []
    cy = []
    regions = regionprops(label)
    for props in regions:
        vals.append(props.label)
        y0, x0 = props.centroid
        cx.append(int(x0))
        cy.append(int(y0))
    return vals, cx, cy


def save_label(hdfpath, fov, frame, label):
    """
    Save label at field of view (fov), frame (frame) in the h5py file (hdfpath).
    h5 file has group fov_i and dataset frame_i.
    Labels are np.uint16 then scaled to 255, chosen to be easily converted to RGB color map.
    """
    file = h5py.File(hdfpath, "r+")
    if f"frame_{frame}" in file[f"/fov_{fov}"]:
        dataset = file[f"/fov_{fov}/frame_{frame}"]
        dataset[:] = label.astype(np.uint16)
    else:
        file.create_dataset(f"/fov_{fov}/frame_{frame}", data=label.astype(np.uint16), compression="gzip")
    file.close()


def save_unet_seg_result(hdfpath, fov, frame, label):
    """
    Save label at unet/field of view (fov)/frame (frame) in the h5py file (hdfpath).
    h5 file now has unet group and group fov_i and dataset frame_i.
    Labels are np.uint16 then scaled to 255, chosen to be easily converted to RGB color map.
    """
    file = h5py.File(hdfpath, "r+")
    if "/unet" not in file:
        file.create_group("/unet")
    if f"/unet/fov_{fov}" not in file["/unet"]:
        file.create_group(f"/unet/fov_{fov}")
    if f"frame_{frame}" in file[f"/unet/fov_{fov}"]:
        dataset = file[f"/unet/fov_{fov}/frame_{frame}"]
        dataset[:] = label.astype(np.uint16)
    else:
        file.create_dataset(f"/unet/fov_{fov}/frame_{frame}", data=label.astype(np.uint16), compression="gzip")
    file.close()


def get_default_path(nd2filepath, post: str):
    """
    Generate default file path from the input nd2filepath.
    e.g. if nd2 file is experiment.nd2, post is h5,
    then return experiment.h5.
    The same default file path is used to save label automatically.
    """
    temp_list = nd2filepath.split('/')
    tmp = ""
    for k in range(0, len(temp_list) - 1):
        tmp += temp_list[k] + '/'
    hdf_name = temp_list[-1].split('.')[-2]
    return tmp + hdf_name + post


def get_label_from_hdf(hdfpath, fov, frame) -> np.ndarray:
    """
    Get the label for current frame and current fov.
    Return None if label is not found in the hdf file.
    """
    # fov and time are stored as fov group and frame dataset as /fov_i/frame_j
    file = h5py.File(hdfpath, "r")
    if f"frame_{frame}" in file[f"/fov_{fov}"]:
        label = file[f"/fov_{fov}/frame_{frame}"][:]
    else:
        label = None
    file.close()
    return label


def get_seg_result_from_hdf(hdfpath, fov, frame):
    """
    Get the label for current frame and current fov.
    Return None if label is not found in the hdf file.
    """
    # fov and time are stored as fov group and frame dataset as /fov_i/frame_j
    file = h5py.File(hdfpath, "r")
    if f"/unet" not in file or f"/unet/fov_{fov}" not in file["/unet"]:
        return None
    if f"frame_{frame}" in file[f"/unet/fov_{fov}"]:
        label = file[f"/unet/fov_{fov}/frame_{frame}"][:]
    else:
        label = None
    file.close()
    return label


def determine_mother_daughter_relation(df):
    if "Mother" not in df.columns:
        df["Mother"] = pd.Series(dtype="object")
    fovs = np.unique(df["Fov"])
    for fov in fovs:
        df_fov = df[df["Fov"] == fov]
        labels = np.unique(df[df["Fov"] == fov]["Label"])
        for i, lv in enumerate(labels):
            first_frame = df_fov.loc[df_fov["Label"] == lv, "Frame"].min()
            df_frame = df_fov.loc[df_fov["Frame"] == first_frame, :]
            df_frame = df_frame.set_index("Label")
            x0 = df_frame.loc[lv, "Centroid_x"]
            y0 = df_frame.loc[lv, "Centroid_y"]

            lv_dist_matrix = np.zeros((len(df_frame.index), 2))
            # collect label value distance matrix
            for j, idx in enumerate(df_frame.index):
                # self-self distance set to a large number
                lv_dist_matrix[j, 0] = idx  # cell label value
                lv_dist_matrix[j, 1] = 1e5  # distance
                if idx != lv:
                    x = df_frame.loc[idx, "Centroid_x"]
                    y = df_frame.loc[idx, "Centroid_y"]
                    # calculate distance of center to cell segment bounds
                    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - df_frame.loc[idx, "Length_major"]
                    lv_dist_matrix[j, 1] = dist
            sort_idx = np.argsort(lv_dist_matrix[:, 1])
            # assign to a mother
            found = False
            area0 = df_frame.loc[lv, "Area"]
            for idx in sort_idx:
                mother_id = lv_dist_matrix[idx, 0]
                area1 = df_frame.loc[mother_id, "Area"]
                if lv_dist_matrix[idx, 1] > 20:
                    break
                elif area1 < area0 * 2:
                    continue
                else:
                    found = True
                    df.loc[df["Label"] == lv, "Mother"] = mother_id
                    break
            # if mother is not found, assign mother to itself
            if not found:
                df.loc[df["Label"] == lv, "Mother"] = lv

# def export_movie(self):
#    # export movie of pictures of current channel and fov
#    # labels are turned into rgb
#    file, _ = QFileDialog.getSaveFileName(self, "Save movie", ".\\", "mp4 file (*.mp4)")
#    if not file:
#        return

#    self.progress_bar.setMaximum(self.total_frames)
#    self.progress_bar.show()

#    def make_frame(t):
#        t = int(t * 4)
#        self.progress_bar.setValue(t)
#        color_image = np.zeros(self.image_shape)
#        if t < self.total_frames:
#            image = self.get_image(t)
#            label = get_label_from_hdf(self.hdfpath, self.fov, t)
#            if label is None:
#                return
#            gray_image = img_as_float(image)
#            color_image = skimage.color.gray2rgb(gray_image)
#            for lv in np.unique(label):
#                if lv == 0:
#                    continue
#                else:
#                    q_color = QColor(get_label_color(lv))
#                    r = q_color.red()
#                    g = q_color.green()
#                    b = q_color.blue()
#                    # multiplier = np.array([r, g, b]) / 255
#                    color_image[label == lv, 0] *= r / 255
#                    color_image[label == lv, 1] *= g / 255
#                    color_image[label == lv, 2] *= b / 255

#        low = 0.2 * 255
#        high = 0.9 * 255
#        img = skimage.exposure.rescale_intensity(color_image, out_range=(low, high))
#        return img
#
#    animation = moviepy.editor.VideoClip(make_frame, duration=self.total_frames / 4)
#    animation.write_videofile(file, fps=4)
#    self.progress_bar.hide()
#    QMessageBox.information(self, "Info", "Movie ready.", QMessageBox.Ok, QMessageBox.Ok)
#    # animation.write_gif("time-lapse.gif", fps=24)
