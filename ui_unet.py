# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'unet.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(492, 618)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_range_label = QtWidgets.QLabel(Dialog)
        self.frame_range_label.setObjectName("frame_range_label")
        self.horizontalLayout_3.addWidget(self.frame_range_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setMinimumSize(QtCore.QSize(150, 0))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.start_frame_line = QtWidgets.QLineEdit(Dialog)
        self.start_frame_line.setMaximumSize(QtCore.QSize(100, 16777215))
        self.start_frame_line.setObjectName("start_frame_line")
        self.horizontalLayout.addWidget(self.start_frame_line)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setMinimumSize(QtCore.QSize(150, 0))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.end_frame_line = QtWidgets.QLineEdit(Dialog)
        self.end_frame_line.setMaximumSize(QtCore.QSize(100, 16777215))
        self.end_frame_line.setObjectName("end_frame_line")
        self.horizontalLayout_2.addWidget(self.end_frame_line)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.select_all_box = QtWidgets.QCheckBox(Dialog)
        self.select_all_box.setObjectName("select_all_box")
        self.verticalLayout_2.addWidget(self.select_all_box)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.fov_list_widget = QtWidgets.QListWidget(Dialog)
        self.fov_list_widget.setObjectName("fov_list_widget")
        self.horizontalLayout_4.addWidget(self.fov_list_widget)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_4 = QtWidgets.QLabel(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(150, 0))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.threshold_line = QtWidgets.QLineEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_line.sizePolicy().hasHeightForWidth())
        self.threshold_line.setSizePolicy(sizePolicy)
        self.threshold_line.setObjectName("threshold_line")
        self.horizontalLayout_5.addWidget(self.threshold_line)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setMinimumSize(QtCore.QSize(150, 0))
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.seed_dist_line = QtWidgets.QLineEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.seed_dist_line.sizePolicy().hasHeightForWidth())
        self.seed_dist_line.setSizePolicy(sizePolicy)
        self.seed_dist_line.setObjectName("seed_dist_line")
        self.horizontalLayout_6.addWidget(self.seed_dist_line)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_7.addWidget(self.label_6)
        self.weight_path_line = QtWidgets.QLineEdit(Dialog)
        self.weight_path_line.setObjectName("weight_path_line")
        self.horizontalLayout_7.addWidget(self.weight_path_line)
        self.chose_weight_button = QtWidgets.QPushButton(Dialog)
        self.chose_weight_button.setObjectName("chose_weight_button")
        self.horizontalLayout_7.addWidget(self.chose_weight_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.start_run_button = QtWidgets.QPushButton(Dialog)
        self.start_run_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.start_run_button.setObjectName("start_run_button")
        self.horizontalLayout_8.addWidget(self.start_run_button)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem6)
        self.close_button = QtWidgets.QPushButton(Dialog)
        self.close_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.close_button.setObjectName("close_button")
        self.horizontalLayout_8.addWidget(self.close_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.horizontalLayout_9.addLayout(self.verticalLayout_3)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "UNet"))
        self.frame_range_label.setToolTip(_translate("Dialog", "Input range, larger or smaller values will be clipped"))
        self.frame_range_label.setText(_translate("Dialog", "Frame range (1-144)"))
        self.label.setText(_translate("Dialog", "Start frame"))
        self.start_frame_line.setToolTip(_translate("Dialog", "Start from 0"))
        self.start_frame_line.setText(_translate("Dialog", "0"))
        self.label_2.setText(_translate("Dialog", "End frame"))
        self.end_frame_line.setToolTip(_translate("Dialog", "End number is not included"))
        self.end_frame_line.setText(_translate("Dialog", "144"))
        self.label_3.setText(_translate("Dialog", "Field(s) of view"))
        self.select_all_box.setText(_translate("Dialog", "select all"))
        self.label_4.setText(_translate("Dialog", "Threshold value"))
        self.threshold_line.setText(_translate("Dialog", "0.5"))
        self.label_5.setText(_translate("Dialog", "Minimum seed distance"))
        self.seed_dist_line.setText(_translate("Dialog", "5"))
        self.label_6.setText(_translate("Dialog", "Weights"))
        self.chose_weight_button.setText(_translate("Dialog", "Chose"))
        self.start_run_button.setText(_translate("Dialog", "Start run"))
        self.close_button.setText(_translate("Dialog", "Close"))