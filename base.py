import sys
from skimage.measure import regionprops
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
    Labels are np.uint8 with maximum 255, chosen to be easily converted to RGB color map.
    """
    file = h5py.File(hdfpath, "r+")
    if f"frame_{frame}" in file[f"/fov_{fov}"]:
        dataset = file[f"/fov_{fov}/frame_{frame}"]
        dataset[:] = label.astype(np.uint16)
    else:
        file.create_dataset(f"/fov_{fov}/frame_{frame}", data=label.astype(np.uint16), compression="gzip")
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
