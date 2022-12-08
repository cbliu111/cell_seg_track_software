import sys
import uuid
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage, qRgb
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QRunnable, QObject, pyqtSignal as Signal, pyqtSlot as Slot, QThreadPool, QPointF, QRect, QPoint, QRectF, QSize
from PIL import Image, ImageDraw
from base import numpy_to_image, COLOR_TABLE


class ViewRenderSignals(QObject):
    """
    Signal finish of colored label pixmap rendering.
    The pixmap is transfered through the signal.
    """
    render_finished = Signal(QPixmap)


class ImageViewRender(QRunnable):
    """
    Change label according to input brush parameters,
    and render a QPixmap for displaying in LabelWidget.
    Only label is used for coloring, tried skimage label2rgb, not fast enough.
    Line mask is not implemented and will be removed.
    """

    def __init__(self):
        super(ImageViewRender, self).__init__()
        self.w, self.h = 512, 512
        self.worker_id = uuid.uuid4().hex
        self._show_label = False
        self.signals = ViewRenderSignals()

    @Slot()
    def run(self):
        if self._show_label:
            q_image = numpy_to_image(self.label.astype(np.uint8), QImage.Format_Indexed8)
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

