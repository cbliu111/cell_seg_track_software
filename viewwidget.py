import sys
import numpy as np
import skimage.exposure
from PyQt5.QtCore import QObject, QTimer, Qt, QThreadPool, pyqtSignal as Signal, pyqtSlot as Slot, QPoint, QPointF
from PyQt5.QtWidgets import QWidget, QApplication, QMenu
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QKeyEvent, QPen, QMouseEvent, QFont, QCursor
from imageviewrender import ImageViewRender
from base import numpy_to_image, get_label_centers, DEFAULT_COLORS


class ViewWidget(QWidget):
    """
    Use to view of previous and next frame at the same window
    """
    send_zoom_point = Signal(QPointF)

    def __init__(self, parent=None, input_image=None, input_label=None):
        super(ViewWidget, self).__init__()
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.ArrowCursor)
        self.setFixedSize(512, 512)
        self.setMouseTracking(True)

        if input_image is None:
            self.image = np.zeros((512, 512), dtype=np.uint16)
        else:
            self.image = input_image
        if input_label is None:
            self.label = np.zeros_like(self.image, dtype=np.uint8)
        else:
            self.label = input_label
        q_image = numpy_to_image(self.image, QImage.Format_Grayscale16)
        q_label = numpy_to_image(self.label, QImage.Format_Indexed8)
        self.pix_image = QPixmap(q_image)
        self.pix_label = QPixmap(q_label)

        self.w, self.h = self.image.shape

        self.scaled_image = None
        self.scaled_label = None
        self.scale = 1

        self.offset = QPointF()
        self.mouse_pos = QPointF()
        self.zoom_point = QPointF()
        self.zoom_factor = 5
        self.scroll_factor = 30
        self.coordinate = ""

        # label id
        self.id_values = []
        self.id_x_cs = []
        self.id_y_cs = []

        self.leave = False
        self._show_label = True
        self.show_id = False
        self.label_value = 0

        # define render
        self.render = ImageViewRender()
        self.render.signals.render_finished.connect(self.update_label_pixmap)
        self.render.label = self.label
        self.threadpool = QThreadPool()
        self.threadpool.start(self.render)

    def paintEvent(self, event):

        nw = int(self.width() * self.scale)
        nh = int(self.height() * self.scale)
        self.offset = self.zoom_point * (1 - self.scale)
        self.scaled_image = self.pix_image.scaled(nw, nh, Qt.KeepAspectRatio)
        if self._show_label:
            self.scaled_label = self.pix_label.scaled(nw, nh, Qt.KeepAspectRatio)

        painter = QPainter(self)
        painter.setOpacity(1)
        painter.drawPixmap(self.offset, self.scaled_image)
        if self._show_label:
            painter.setOpacity(0.2)
            painter.drawPixmap(self.offset, self.scaled_label)
        if self.show_id and self._show_label:
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setOpacity(1)
            painter.setPen(QPen(Qt.green))
            for i, v in enumerate(self.id_values):
                p = self.map_to_screen(QPoint(self.id_x_cs[i], self.id_y_cs[i]))
                painter.drawText(p, f"{v}")

        if self.leave:
            return

        painter.end()

    def set_image(self, input_image):
        self.w, self.h = self.image.shape
        low = 0.3 * 65535
        high = 0.9 * 65535
        img = skimage.exposure.rescale_intensity(input_image, out_range=(low, high))
        self.image = img.astype(np.uint16)
        # self.image = skimage.exposure.adjust_gamma(input_image, 0.4)
        q_image = numpy_to_image(self.image, QImage.Format_Grayscale16)
        self.pix_image = QPixmap(q_image)
        self.update()

    def set_label(self, input_label):
        self.label = input_label
        self.render.label = input_label
        self.show_image()

    def set_scale(self, s):
        self.scale = s

    def map_from_screen(self, p: QPointF):
        """
        Return the image coordinate from the point on the screen.
        The current screen has the same resolution with the scaled image.
        Thus scaling of the mouse position is not right.
        Image rows corresponding to height, and also to y.
        Image cols corresponding to width, and also to x.
        """
        x = (p.x() - self.offset.x()) / self.scaled_image.width() * self.image.shape[1]
        y = (p.y() - self.offset.y()) / self.scaled_image.height() * self.image.shape[0]
        img_pos_x = int(x)
        img_pos_y = int(y)
        return QPoint(img_pos_y, img_pos_x)

    def map_to_screen(self, p: QPoint):
        """
        The reverse function of map_from_screen.
        """
        x = p.x() / self.image.shape[1] * self.scaled_image.width() + self.offset.x()
        y = p.y() / self.image.shape[0] * self.scaled_image.height() + self.offset.y()
        return QPointF(x, y)

    def show_image(self):
        self.render.run()
        self.update()

    def zoom(self, sf):
        self.scale = sf
        self.update()

    def zoom_in(self):
        self.scale = self.zoom_factor
        self.update()

    def zoom_out(self):
        self.scale = 1
        self.update()

    def scroll_left(self):
        x = self.zoom_point.x()
        self.zoom_point.setX(x + self.scroll_factor)
        self.update()

    def scroll_right(self):
        x = self.zoom_point.x()
        self.zoom_point.setX(x - self.scroll_factor)
        self.update()

    def scroll_up(self):
        y = self.zoom_point.y()
        self.zoom_point.setY(y + self.scroll_factor)
        self.update()

    def scroll_down(self):
        y = self.zoom_point.y()
        self.zoom_point.setY(y - self.scroll_factor)
        self.update()

    def set_zoom_factor(self, zf=5):
        """
        set the current zoom folds
        larger zoom folds will significantly slow down the image refresh speed
        :param zf: zoom factor, default value is 5
        :return: None
        """
        self.scale = 1
        self.zoom_factor = zf
        self.update()

    @Slot(QPixmap)
    def update_label_pixmap(self, pixmap):
        self.id_values, self.id_x_cs, self.id_y_cs = get_label_centers(self.render.label)
        self.pix_label = pixmap
        self.update()

    def hide_label(self):
        self._show_label = False
        self.render._show_label = False
        self.update()

    def show_label(self):
        self._show_label = True
        self.render._show_label = True
        self.update()

    def show_label_id(self):
        self.show_id = True
        self.update()

    def hide_label_id(self):
        self.show_id = False
        self.update()

    def leaveEvent(self, event):
        self.leave = True
        self.update()

    def enterEvent(self, event):
        self.leave = False
        self.setFocus()
        self.update()

    def keyPressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()


if __name__ == "__main__":
    from skimage import io

    app = QApplication(sys.argv)
    image = io.imread("cell.tif")
    label = np.zeros(image.shape, dtype=np.uint8)
    label[0:200, 0:100] = 1
    label[100:150, 100:200] = 2
    label[200:250, 200:300] = 3
    label[300:350, 300:400] = 4
    label[400:450, 400:500] = 5
    win = ViewWidget(None, image, label)
    win.show()
    win.set_zoom_factor(5)
    win.show_label()
    win.show_label_id()
    win.show_image()
    sys.exit(app.exec())
