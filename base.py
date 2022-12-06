import sys
from sklearn.decomposition import PCA
from skimage.measure import regionprops
import numpy as np
import h5py
from PyQt5.QtGui import qRgb, QColor, QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QPushButton

DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                  'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')


def get_label_color(lv):
    return DEFAULT_COLORS[(lv - 1) % len(DEFAULT_COLORS)]


COLOR_TABLE = [qRgb(0, 0, 0)] + [QColor(i).rgb() for i in DEFAULT_COLORS] * 20


def numpy_to_image(m: np.ndarray, fmt: QImage.Format):
    return QImage(m.data, m.shape[0], m.shape[1], m.strides[0], fmt)


def set_button_color(button: QPushButton, r, g, b):
    s = f"background-color: rgb({r}, {g}, {b})"
    button.setStyleSheet(s)


def numpy_to_color(m: np.ndarray, r, g, b):
    q_image = numpy_to_image(m, QImage.Format_Grayscale16)
    w = m.shape[0]
    h = m.shape[1]
    for i in range(w):
        for j in range(h):
            q_image.setPixel(i, j, qRgb(r, g, b))
    return q_image


def label_statistics(image, label):
    """Calculate statistics about cells. Passing None to image will
    create dictionary to zeros, which allows to extract dictionary keys"""
    area = 0
    if image is not None:
        cell_vals = image[label]
        area = label.sum()
        tot_intensity = cell_vals.sum()
        mean = tot_intensity / area if area > 0 else 0
        var = np.var(cell_vals)

        # Center of mass
        y, x = label.nonzero()
        com_x = np.mean(x)
        com_y = np.mean(y)

        # PCA only works for multiple points
        if area > 1:
            pca = PCA().fit(np.array([x, y]).T)
            pc1_x, pc1_y = pca.components_[0, :]
            angle = np.arctan(pc1_y / pc1_x) / (2 * np.pi) * 360
            v1, v2 = pca.explained_variance_

            len_maj = 4 * np.sqrt(v1)
            len_min = 4 * np.sqrt(v2)
        else:
            angle = 0
            len_maj = 1
            len_min = 1

    else:
        mean = 0
        var = np.nan
        tot_intensity = 0
        com_x = np.nan
        com_y = np.nan
        angle = np.nan
        len_maj = np.nan
        len_min = np.nan

    return {'Area': area,
            'Mean': mean,
            'Variance': var,
            'Total Intensity': tot_intensity,
            'Center of Mass X': com_x,
            'Center of Mass Y': com_y,
            'Angle of Major Axis': angle,
            'Length Major Axis': len_maj,
            'Length Minor Axis': len_min}


def get_label_centers(label: np.ndarray):
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


def save_label(hdfpath, fov, frame, img):
    # h5 file has group fov_i and dataset frame_i
    # labels are np.uint8
    file = h5py.File(hdfpath, "r+")
    if f"frame_{frame}" in file[f"/fov_{fov}"]:
        dataset = file[f"/fov_{fov}/frame_{frame}"]
        dataset[:] = img
    else:
        file.create_dataset(f"/fov_{fov}/frame_{frame}", data=img, compression="gzip")
    file.close()


def get_default_path(nd2filepath, post: str):
    temp_list = nd2filepath.split('/')
    tmp = ""
    for k in range(0, len(temp_list) - 1):
        tmp += temp_list[k] + '/'
    hdf_name = temp_list[-1].split('.')[-2]
    return tmp + hdf_name + post
