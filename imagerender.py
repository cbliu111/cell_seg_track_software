import sys
import uuid
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage, qRgb
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QRunnable, QObject, pyqtSignal as Signal, pyqtSlot as Slot, QThreadPool, QPointF, QRect, QPoint, QRectF, QSize
from PIL import Image, ImageDraw
from base import numpy_to_image, COLOR_TABLE


class RenderSignals(QObject):
    """
    Signal finish of colored label pixmap rendering.
    The pixmap is transfered through the signal.
    """
    render_finished = Signal(QPixmap)


class ImageRender(QRunnable):
    """
    Change label according to input brush parameters,
    and render a QPixmap for displaying in LabelWidget.
    Only label is used for coloring, tried skimage label2rgb, not fast enough.
    Line mask is not implemented and will be removed.
    """

    def __init__(self):
        super(ImageRender, self).__init__()
        self.w, self.h = 512, 512
        self.worker_id = uuid.uuid4().hex
        self.brush_size = 50
        self.label_value = 0
        self.scale = 1
        self.radius = np.ceil(self.brush_size / 2 / self.scale)
        self.label = np.zeros((self.w, self.h), dtype=np.uint8)
        self.mask = np.zeros((self.w, self.h), dtype=bool)
        self.xx, self.yy = np.meshgrid(np.arange(self.w), np.arange(self.h))
        self.line = []
        self._show_label = False
        self.brush_type = "circle"
        self.draw_mode = "draw"
        self.brush_mask = {
            "circle": self.circle_mask,
            "square": self.square_mask,
            "polygon": self.polygon_mask,
            "line": self.line_mask,
        }
        self.draw_route = {
            "draw": self.draw,
            "erase": self.erase,
        }
        self.signals = RenderSignals()

    def set_image_shape(self, shape: tuple):
        self.w, self.h = shape
        self.mask = np.zeros((self.w, self.h), dtype=bool)
        self.xx, self.yy = np.meshgrid(np.arange(self.w), np.arange(self.h))

    def set_brush_size(self, size):
        self.brush_size = size
        self.radius = np.ceil(self.brush_size / 2 / self.scale)

    def set_scale(self, scale):
        self.scale = scale
        self.radius = np.ceil(self.brush_size / 2 / self.scale)

    def circle_mask(self):
        self.mask[:] = False
        for p in self.line:
            cx = p.y()
            cy = p.x()
            radius = min(self.radius, self.w - cx, self.h - cy)
            dist = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2)
            indexes = dist <= radius
            self.mask[indexes] = True
        self.line = []

    def square_mask(self):
        self.mask[:] = False
        for p in self.line:
            cx = p.y()
            cy = p.x()
            radius = min(self.radius, self.w - cx, self.h - cy)
            dist_x = np.abs(self.xx - cx)
            dist_y = np.abs(self.yy - cy)
            # outside the square are all False
            ix = dist_x <= radius
            iy = dist_y <= radius
            self.mask[ix * iy] = True
        self.line = []

    def line_mask(self):
        # TODO: what is the point here?
        # self.mask[:] = False
        # x0 = 0
        # y0 = 0
        # for i, p in enumerate(self.line):
        #     if i == 0:
        #         x0 = int(p.y())
        #         y0 = int(p.x())
        #         continue
        #     else:
        #         x1 = int(p.y())
        #         y1 = int(p.x())
        #         xx = self.xx[x0:x1]
        #         yy = self.yy[y0:y1]
        #         d = (xx - x0) * (yy - y0) / (x1 - x0)
        #         dist = (x1 - x0) * (d - (yy - y0)) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        #         indexes = dist <= 3
        #         self.mask[indexes] = True
        # self.line = []
        pass

    def polygon_mask(self):
        self.mask[:] = False
        temp_img = Image.new("L", (self.w, self.h), 0)
        vertexes = []
        for p in self.line:
            cx = int(p.y())
            cy = int(p.x())
            vertexes.append((cx, cy))
        self.line = []
        if len(vertexes) > 1:
            ImageDraw.Draw(temp_img).polygon(vertexes, outline=1, fill=1)
            self.mask = np.array(temp_img).astype(bool)

    def draw(self):
        self.label[self.mask] = self.label_value

    def erase(self):
        self.label[self.mask] = 0

    @Slot()
    def run(self):
        if self._show_label:
            self.brush_mask[self.brush_type]()  # generate brush mask according to brush_mode
            self.draw_route[self.draw_mode]()  # draw or erase value on the label matrix
            q_image = numpy_to_image(self.label, QImage.Format_Indexed8)
            q_image.setColorTable(COLOR_TABLE)
            q_image = q_image.convertToFormat(QImage.Format_RGB888)
            q_mask = numpy_to_image(self.label, QImage.Format_Indexed8)
            pixmap = QPixmap(q_image)
            mask = QPixmap(q_mask)
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawPixmap(0, 0, mask)
            painter.end()
            self.signals.render_finished.emit(pixmap)


if __name__ == "__main__":
    from skimage import io
    class Window(QWidget):
        def __init__(self):
            super(Window, self).__init__()
            self.setMouseTracking(True)
            self.mouse_pos = QPointF()
            self.setFixedSize(500, 500)
            self.setWindowTitle("Draw")

            self.image = io.imread("cell.tif")
            self.label = np.zeros(self.image.shape, dtype=np.uint8)
            self.label[0:200, 0:100] = 1
            self.label[100:150, 100:200] = 2
            self.label[200:250, 200:300] = 3
            self.label[300:350, 300:400] = 4
            self.label[400:450, 400:500] = 5

            self.offset = QPointF()
            self.coordinate = ""
            self.brush_traj = [] # plot brush trajectory on pixmap layer
            self.line = [] # used to draw on label layer
            self.brush_size = 50
            self.zoom_point = QPointF()

            q_image = numpy_to_image(self.image, QImage.Format_Grayscale16)
            q_label = numpy_to_image(self.image, QImage.Format_Indexed8)
            self.pix_image = QPixmap(q_image)
            self.pix_label = QPixmap(q_label)
            self.scaled_image = None
            self.scaled_label = None
            self.scale = 1
            self.render = ImageRender()
            self.render.signals.render_finished.connect(self.update_label_pixmap)
            self.render.set_image_shape(self.image.shape)
            self.render.set_brush_size(self.brush_size)
            self.render.label = self.label
            self.threadpool = QThreadPool()
            self.threadpool.start(self.render)
            self.show()

        def paintEvent(self, event):

            nw = int(self.width() * self.scale)
            nh = int(self.height() * self.scale)
            self.offset = self.zoom_point * (1 - self.scale)
            self.scaled_image = self.pix_image.scaled(nw, nh, Qt.KeepAspectRatio)
            self.scaled_label = self.pix_label.scaled(nw, nh, Qt.KeepAspectRatio)

            painter = QPainter(self)
            painter.setOpacity(1)
            painter.drawPixmap(self.offset, self.scaled_image)
            painter.setOpacity(0.2)
            painter.drawPixmap(self.offset, self.scaled_label)

            # show mouse position
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(self.coordinate)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 127))
            painter.drawRect(int((self.width()-text_width)/2-5), 0, text_width+10, metrics.lineSpacing()+5)
            painter.setPen(Qt.red)
            painter.drawText(int((self.width()-text_width)/2), metrics.leading()+metrics.ascent(), self.coordinate)

            bx = self.mouse_pos.x() - self.brush_size / 2
            by = self.mouse_pos.y() - self.brush_size / 2
            painter.drawEllipse(int(bx), int(by), self.brush_size, self.brush_size)
            for p in self.brush_traj:
                bx = p.x() - self.brush_size / 2
                by = p.y() - self.brush_size / 2
                painter.drawEllipse(int(bx), int(by), self.brush_size, self.brush_size)
            painter.end()

        def map_from_screen(self, p: QPointF):
            """return the image coordinate from the point on the screen"""
            # the current screen has the same resolution with the scaled image
            # thus scaling of the mouse position is not right
            # image rows corresponding to height, and also to y
            # image cols corresponding to width, and also to x
            x = (p.x() - self.offset.x()) / self.scaled_image.width() * self.image.shape[1]
            y = (p.y() - self.offset.y()) / self.scaled_image.height() * self.image.shape[0]
            img_pos_x = int(x)
            img_pos_y = int(y)
            return QPoint(img_pos_y, img_pos_x)

        def mouseMoveEvent(self, event):
            self.mouse_pos = event.pos()
            image_pos = self.map_from_screen(self.mouse_pos)
            self.coordinate = f"xy[{image_pos.x()}, {image_pos.y()}], label[{self.label[image_pos.x(), image_pos.y()]}]"
            if event.buttons() & Qt.LeftButton:
                p = self.map_from_screen(event.pos())
                self.line.append(p)
                self.brush_traj.append(event.pos())
            self.update()

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                p = self.map_from_screen(event.pos())
                self.line.append(p)
                self.update()

        def mouseReleaseEvent(self, event):
            self.render.line = self.line
            self.render.label_value = 1
            self.render.run()
            self.brush_traj = []
            self.line = []
            self.update()

        def keyPressEvent(self, event):
            # zoom picture
            if event.key() == Qt.Key_Z:
                sf = self.scale
                sf = 1 if sf != 1 else 5
                self.scale = sf
                self.render.set_scale(sf)
                self.zoom_point = self.mouse_pos
            self.update()

        def update_label_pixmap(self, pixmap):
            self.pix_label = pixmap
            self.update()

    app = QApplication(sys.argv)
    win = Window()
    sys.exit(app.exec())
