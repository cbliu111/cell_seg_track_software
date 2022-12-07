import sys
import numpy as np
import skimage.exposure
from PyQt5.QtCore import QObject, QTimer, Qt, QThreadPool, pyqtSignal as Signal, pyqtSlot as Slot, QPoint, QPointF
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QKeyEvent, QPen, QMouseEvent, QFont
from imagerender import ImageRender
from base import numpy_to_image, get_label_centers, DEFAULT_COLORS

class LabelWidget(QWidget):
    read_next_frame = Signal()
    read_previous_frame = Signal()
    label_updated = Signal()
    select_id_at_mouse_position = Signal(int, int)
    copy_id_at_mouse_position = Signal(int, int)
    paste_id_inplace = Signal()
    delete_id_at_mouse_position = Signal(int, int)
    draw_at_mouse_position = Signal(int, int)

    def __init__(self, parent=None, input_image=None, input_label=None):
        super(LabelWidget, self).__init__()
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        self.setFixedSize(500, 500)

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
        # plot brush trajectory on pixmap layer
        self.brush_traj = []
        # used to draw on label layer
        self.line = []
        self.brush_size = 50
        self.coordinate = ""

        # label id
        self.id_values = []
        self.id_x_cs = []
        self.id_y_cs = []

        self.leave = False
        self._show_label = True
        self.show_id = False
        self.label_value = 0
        self.brush_type = "circle"
        self.draw_mode = "draw"
        self.brush_route = {
            "circle": self.circle_brush,
            "square": self.square_brush,
            "polygon": self.polygon_brush,
            "line": self.line_brush,
        }
        """Keys:
                D: draw mode
                E: erase mode
                Z: zoom out or store origin scale
                Left: scroll image to west
                right: scroll image to east
                down: scroll image to south
                up: scroll image to north
                F: go to next frame
                A: go back to previous frame
                C: copy label at where the mouse hold
                V: paste copied label for e.g. in other frame at the same location
                X: delete label at where the mouse hold"""
        self.key_route = {
            Qt.Key_D: self.draw,
            Qt.Key_E: self.erase,
            Qt.Key_Z: self.zoom,
            Qt.Key_Left: self.scroll_left,
            Qt.Key_Right: self.scroll_right,
            Qt.Key_Down: self.scroll_down,
            Qt.Key_Up: self.scroll_up,
            Qt.Key_F: self.next_frame,
            Qt.Key_A: self.previous_frame,
            Qt.Key_C: self.copy_id,
            Qt.Key_V: self.paste_id,
            Qt.Key_X: self.delete_id,
            Qt.Key_S: self.select_id,
        }
        # define render
        self.render = ImageRender()
        self.render.signals.render_finished.connect(self.update_label_pixmap)
        self.render.set_image_shape(self.image.shape)
        self.render.set_brush_size(self.brush_size)
        self.render.set_scale(self.scale)
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
        # show mouse position
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setOpacity(1)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.coordinate)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 127))
        painter.drawRect(int((self.width() - text_width) / 2 - 5), 0, text_width + 10, metrics.lineSpacing() + 5)
        painter.setPen(Qt.red)
        painter.drawText(int((self.width() - text_width) / 2), metrics.leading() + metrics.ascent(), self.coordinate)

        if self._show_label:
            # draw brush trajectory
            painter.setOpacity(0.2)
            painter.setPen(Qt.red)
            self.brush_route[self.brush_type](painter)

        painter.end()

    def get_brush_color(self):
        if self.draw_mode == "draw":
            # c = COLORS[(self.label_value-1) % len(COLORS)]
            c = DEFAULT_COLORS[(self.label_value-1) % len(DEFAULT_COLORS)]
            return QColor(c)
        else:
            return QColor(255, 255, 255)

    def circle_brush(self, painter):
        bx = self.mouse_pos.x() - self.brush_size / 2
        by = self.mouse_pos.y() - self.brush_size / 2
        color = self.get_brush_color()
        painter.setBrush(color)
        painter.drawEllipse(int(bx), int(by), self.brush_size, self.brush_size)
        for p in self.brush_traj:
            bx = p.x() - self.brush_size / 2
            by = p.y() - self.brush_size / 2
            painter.drawEllipse(int(bx), int(by), self.brush_size, self.brush_size)

    def square_brush(self, painter):
        bx = self.mouse_pos.x() - self.brush_size / 2
        by = self.mouse_pos.y() - self.brush_size / 2
        color = self.get_brush_color()
        painter.setBrush(color)
        painter.drawRect(int(bx), int(by), self.brush_size, self.brush_size)
        for p in self.brush_traj:
            bx = p.x() - self.brush_size / 2
            by = p.y() - self.brush_size / 2
            painter.drawRect(int(bx), int(by), self.brush_size, self.brush_size)

    def polygon_brush(self, painter):
        pen = QPen()
        color = self.get_brush_color()
        pen.setColor(color)
        pen.setWidth(3)
        painter.setPen(pen)
        lp = QPointF()
        for i, p in enumerate(self.brush_traj):
            if i == 0:
                painter.drawEllipse(p, 2, 2)
                lp = p
            else:
                painter.drawLine(lp, p)
                lp = p
        if self.brush_traj:
            painter.drawLine(self.brush_traj[-1], self.brush_traj[0])

    def line_brush(self, painter):
        pen = QPen()
        color = self.get_brush_color()
        pen.setColor(color)
        pen.setWidth(3)
        painter.setPen(pen)
        lp = QPointF()
        for i, p in enumerate(self.brush_traj):
            if i == 0:
                lp = p
                continue
            else:
                painter.drawLine(lp, p)
                lp = p

    def set_image(self, input_image):
        self.w, self.h = self.image.shape
        self.image = skimage.exposure.adjust_gamma(input_image, 0.5)
        self.render.set_image_shape(self.image.shape)
        q_image = numpy_to_image(self.image, QImage.Format_Grayscale16)
        self.pix_image = QPixmap(q_image)
        self.update()

    def set_label(self, input_label):
        self.label = input_label
        self.render.label = input_label
        self.label_value = 0
        self.render.label_value = 0
        self.show_image()

    def set_scale(self, s):
        self.scale = s

    def next_frame(self):
        self.read_next_frame.emit()

    def previous_frame(self):
        self.read_previous_frame.emit()

    def copy_id(self):
        x, y = self.get_image_position_from_mouse()
        self.copy_id_at_mouse_position.emit(x, y)

    def paste_id(self):
        self.paste_id_inplace.emit()

    def delete_id(self):
        x, y = self.get_image_position_from_mouse()
        self.delete_id_at_mouse_position.emit(x, y)

    def select_id(self):
        x, y = self.get_image_position_from_mouse()
        self.select_id_at_mouse_position.emit(x, y)

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

    def map_to_screen(self, p: QPoint):
        x = p.x() / self.image.shape[1] * self.scaled_image.width() + self.offset.x()
        y = p.y() / self.image.shape[0] * self.scaled_image.height() + self.offset.y()
        return QPointF(x, y)

    def get_image_position_from_mouse(self):
        image_pos = self.map_from_screen(self.mouse_pos)
        x = image_pos.x() if image_pos.x() < self.w else self.w
        y = image_pos.y() if image_pos.y() < self.h else self.h
        return x, y

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        x, y = self.get_image_position_from_mouse()
        self.coordinate = f"xy[{x}, {y}], label[{self.label[x, y]}]"
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
        self.render.label_value = self.label_value
        self.brush_traj = []
        self.line = []
        self.show_image()

    def show_image(self):
        self.render.run()
        self.update()

    def zoom(self):
        sf = self.scale
        sf = 1 if sf != 1 else self.zoom_factor
        self.scale = sf
        self.render.set_scale(sf)
        self.zoom_point = self.mouse_pos
        self.update()

    def scroll_left(self):
        x = self.zoom_point.x()
        self.zoom_point.setX(x + 10)
        self.update()

    def scroll_right(self):
        x = self.zoom_point.x()
        self.zoom_point.setX(x - 10)
        self.update()

    def scroll_up(self):
        y = self.zoom_point.y()
        self.zoom_point.setY(y + 10)
        self.update()

    def scroll_down(self):
        y = self.zoom_point.y()
        self.zoom_point.setY(y - 10)
        self.update()

    def set_zoom_factor(self, zf):
        self.scale = 1
        self.zoom_factor = zf
        self.update()

    def draw(self):
        """change mode to draw
        if mouse is hanging over a label, set label value correspondingly
        otherwise create a new label value"""
        self.draw_mode = "draw"
        self.render.draw_mode = "draw"
        x, y = self.get_image_position_from_mouse()
        self.update()
        self.draw_at_mouse_position.emit(x, y)

    def erase(self):
        self.draw_mode = "erase"
        self.render.draw_mode = "erase"
        self.label_value = 0
        self.update()

    def keyPressEvent(self, event):
        if event.key() in self.key_route.keys():
            self.key_route[event.key()]()

    @Slot(QPixmap)
    def update_label_pixmap(self, pixmap):
        self.id_values, self.id_x_cs, self.id_y_cs = get_label_centers(self.render.label)
        self.pix_label = pixmap
        self.update()
        self.label_updated.emit()

    def set_label_value(self, lv):
        self.label_value = lv
        self.render.label_value = lv
        if lv != 0:
            self.draw_mode = "draw"
            self.render.draw_mode = "draw"
        else:
            self.draw_mode = "erase"
            self.render.draw_mode = "erase"
        self.show_image()

    def set_brush_size(self, size: int):
        s = np.clip(size, 10, 100)
        self.brush_size = s
        self.render.set_brush_size(s)
        self.update()

    def set_brush_type(self, t: str):
        self.brush_type = t
        self.render.brush_type = t
        self.update()

    def set_draw(self):
        self.draw_mode = "draw"
        self.render.draw_mode = "draw"
        self.update()

    def set_erase(self):
        self.draw_mode = "erase"
        self.render.draw_mode = "erase"
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
    win = LabelWidget(None, image, label)
    win.show()
    win.set_label_value(3)
    # win.set_label(label)
    # Z is zoom
    # D is draw, default label value is 1
    # E is erase
    win.set_brush_size(30)
    win.set_zoom_factor(10)
    win.show_label()
    win.show_label_id()
    sys.exit(app.exec())
