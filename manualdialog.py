from ui_manualdialog import Ui_ManualDialog
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QAbstractItemView, QPushButton, QDialogButtonBox, \
    QMessageBox


class ManualDialog(QDialog, Ui_ManualDialog):

    def __init__(self, parent=None):
        super(ManualDialog, self).__init__()
        self.setParent(parent)
        self.setupUi(self)

