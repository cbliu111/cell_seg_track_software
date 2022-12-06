import sys
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QSlider, QLabel, QApplication
from PyQt5.QtCore import qRound, pyqtSignal as Signal


class ViewSlider(QSlider):
    value_changed = Signal(int)

    def __init__(self, parent=None):
        super(ViewSlider, self).__init__()
        self.setParent(parent)

    def mousePressEvent(self, event: QMouseEvent):
        a = self.maximum()
        b = self.minimum()
        r = a - b
        pos = qRound(b + (r * (event.x() / self.width())))
        if pos != self.sliderPosition():
            self.setValue(pos)
        self.value_changed.emit(self.value())

    def mouseReleaseEvent(self, event):
        a = self.maximum()
        b = self.minimum()
        r = a - b
        pos = qRound(b + (r * (event.x() / self.width())))
        if pos != self.sliderPosition():
            self.setValue(pos)
        self.value_changed.emit(self.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ViewSlider(None)
    win.show()
    sys.exit(app.exec())