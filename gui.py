# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(988, 710)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_menu = QtWidgets.QWidget(Dialog)
        self.widget_menu.setMinimumSize(QtCore.QSize(245, 0))
        self.widget_menu.setMaximumSize(QtCore.QSize(225, 16777215))
        self.widget_menu.setObjectName("widget_menu")
        self.layout_menu = QtWidgets.QVBoxLayout(self.widget_menu)
        self.layout_menu.setContentsMargins(0, 0, 0, 0)
        self.layout_menu.setObjectName("layout_menu")
        self.scrollArea = QtWidgets.QScrollArea(self.widget_menu)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -21, 226, 610))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_signal_generation = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_signal_generation.setCheckable(True)
        self.pushButton_signal_generation.setChecked(True)
        self.pushButton_signal_generation.setObjectName("pushButton_signal_generation")
        self.verticalLayout_2.addWidget(self.pushButton_signal_generation)
        self.widget_loading_data = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget_loading_data.setObjectName("widget_loading_data")
        self.layout_loading_data = QtWidgets.QVBoxLayout(self.widget_loading_data)
        self.layout_loading_data.setContentsMargins(0, 0, 0, 0)
        self.layout_loading_data.setSpacing(5)
        self.layout_loading_data.setObjectName("layout_loading_data")
        self.widget_signal_generation = QtWidgets.QWidget(self.widget_loading_data)
        self.widget_signal_generation.setObjectName("widget_signal_generation")
        self.layout_signal_generation = QtWidgets.QVBoxLayout(self.widget_signal_generation)
        self.layout_signal_generation.setContentsMargins(0, 0, 0, 0)
        self.layout_signal_generation.setObjectName("layout_signal_generation")
        self.groupBox = QtWidgets.QGroupBox(self.widget_signal_generation)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setContentsMargins(2, 2, 2, 0)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.radioButton_random = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_random.setChecked(True)
        self.radioButton_random.setObjectName("radioButton_random")
        self.verticalLayout_4.addWidget(self.radioButton_random)
        self.radioButton_own = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_own.setObjectName("radioButton_own")
        self.verticalLayout_4.addWidget(self.radioButton_own)
        self.lineEdit_bits = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_bits.setEnabled(True)
        self.lineEdit_bits.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_bits.setObjectName("lineEdit_bits")
        self.verticalLayout_4.addWidget(self.lineEdit_bits)
        self.layout_signal_generation.addWidget(self.groupBox)
        self.groupBox_envelope_parameters = QtWidgets.QGroupBox(self.widget_signal_generation)
        self.groupBox_envelope_parameters.setObjectName("groupBox_envelope_parameters")
        self.layout_envelope_parameters = QtWidgets.QVBoxLayout(self.groupBox_envelope_parameters)
        self.layout_envelope_parameters.setContentsMargins(0, 5, 0, 0)
        self.layout_envelope_parameters.setSpacing(5)
        self.layout_envelope_parameters.setObjectName("layout_envelope_parameters")
        self.widget_carrier_frequency = QtWidgets.QWidget(self.groupBox_envelope_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_carrier_frequency.sizePolicy().hasHeightForWidth())
        self.widget_carrier_frequency.setSizePolicy(sizePolicy)
        self.widget_carrier_frequency.setObjectName("widget_carrier_frequency")
        self.layout_carrier_frequency = QtWidgets.QHBoxLayout(self.widget_carrier_frequency)
        self.layout_carrier_frequency.setContentsMargins(9, 0, 9, -1)
        self.layout_carrier_frequency.setObjectName("layout_carrier_frequency")
        self.label_text_carrier_frequency = QtWidgets.QLabel(self.widget_carrier_frequency)
        self.label_text_carrier_frequency.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_carrier_frequency.setObjectName("label_text_carrier_frequency")
        self.layout_carrier_frequency.addWidget(self.label_text_carrier_frequency)
        self.lineEdit_carrier_frequency = QtWidgets.QLineEdit(self.widget_carrier_frequency)
        self.lineEdit_carrier_frequency.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_carrier_frequency.setObjectName("lineEdit_carrier_frequency")
        self.layout_carrier_frequency.addWidget(self.lineEdit_carrier_frequency)
        self.layout_carrier_frequency.setStretch(0, 1)
        self.layout_carrier_frequency.setStretch(1, 1)
        self.layout_envelope_parameters.addWidget(self.widget_carrier_frequency)
        self.widget_phase = QtWidgets.QWidget(self.groupBox_envelope_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_phase.sizePolicy().hasHeightForWidth())
        self.widget_phase.setSizePolicy(sizePolicy)
        self.widget_phase.setObjectName("widget_phase")
        self.layout_phase = QtWidgets.QHBoxLayout(self.widget_phase)
        self.layout_phase.setContentsMargins(-1, 0, -1, -1)
        self.layout_phase.setObjectName("layout_phase")
        self.label_text_phase = QtWidgets.QLabel(self.widget_phase)
        self.label_text_phase.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_phase.setObjectName("label_text_phase")
        self.layout_phase.addWidget(self.label_text_phase)
        self.lineEdit_phase = QtWidgets.QLineEdit(self.widget_phase)
        self.lineEdit_phase.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_phase.setObjectName("lineEdit_phase")
        self.layout_phase.addWidget(self.lineEdit_phase)
        self.layout_phase.setStretch(0, 1)
        self.layout_phase.setStretch(1, 1)
        self.layout_envelope_parameters.addWidget(self.widget_phase)
        self.layout_signal_generation.addWidget(self.groupBox_envelope_parameters)
        self.pushButton_simulate_signal = QtWidgets.QPushButton(self.widget_signal_generation)
        self.pushButton_simulate_signal.setObjectName("pushButton_simulate_signal")
        self.layout_signal_generation.addWidget(self.pushButton_simulate_signal)
        self.groupBox_noise = QtWidgets.QGroupBox(self.widget_signal_generation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_noise.sizePolicy().hasHeightForWidth())
        self.groupBox_noise.setSizePolicy(sizePolicy)
        self.groupBox_noise.setObjectName("groupBox_noise")
        self.layout_noise = QtWidgets.QVBoxLayout(self.groupBox_noise)
        self.layout_noise.setContentsMargins(0, 5, 0, 0)
        self.layout_noise.setSpacing(5)
        self.layout_noise.setObjectName("layout_noise")
        self.widget_noise = QtWidgets.QWidget(self.groupBox_noise)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_noise.sizePolicy().hasHeightForWidth())
        self.widget_noise.setSizePolicy(sizePolicy)
        self.widget_noise.setObjectName("widget_noise")
        self.layout_noise_text = QtWidgets.QHBoxLayout(self.widget_noise)
        self.layout_noise_text.setContentsMargins(-1, 0, -1, -1)
        self.layout_noise_text.setObjectName("layout_noise_text")
        self.label_text_noise = QtWidgets.QLabel(self.widget_noise)
        self.label_text_noise.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_noise.setObjectName("label_text_noise")
        self.layout_noise_text.addWidget(self.label_text_noise)
        self.lineEdit_noise = QtWidgets.QLineEdit(self.widget_noise)
        self.lineEdit_noise.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_noise.setObjectName("lineEdit_noise")
        self.layout_noise_text.addWidget(self.lineEdit_noise)
        self.layout_noise_text.setStretch(0, 1)
        self.layout_noise_text.setStretch(1, 1)
        self.layout_noise.addWidget(self.widget_noise)
        self.pushButton_add_noise = QtWidgets.QPushButton(self.groupBox_noise)
        self.pushButton_add_noise.setObjectName("pushButton_add_noise")
        self.layout_noise.addWidget(self.pushButton_add_noise)
        self.layout_signal_generation.addWidget(self.groupBox_noise)
        self.layout_loading_data.addWidget(self.widget_signal_generation)
        self.verticalLayout_2.addWidget(self.widget_loading_data)
        self.pushButton_restore_the_original_information = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_restore_the_original_information.setObjectName("pushButton_restore_the_original_information")
        self.verticalLayout_2.addWidget(self.pushButton_restore_the_original_information)
        self.progressBar_recovery = QtWidgets.QProgressBar(self.scrollAreaWidgetContents)
        self.progressBar_recovery.setProperty("value", 0)
        self.progressBar_recovery.setObjectName("progressBar_recovery")
        self.verticalLayout_2.addWidget(self.progressBar_recovery)
        self.widget_execution_time_recovery = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_execution_time_recovery.sizePolicy().hasHeightForWidth())
        self.widget_execution_time_recovery.setSizePolicy(sizePolicy)
        self.widget_execution_time_recovery.setObjectName("widget_execution_time_recovery")
        self.layout_execution_time_recovery = QtWidgets.QHBoxLayout(self.widget_execution_time_recovery)
        self.layout_execution_time_recovery.setContentsMargins(-1, 0, -1, -1)
        self.layout_execution_time_recovery.setObjectName("layout_execution_time_recovery")
        self.label_text_execution_time_recovery = QtWidgets.QLabel(self.widget_execution_time_recovery)
        self.label_text_execution_time_recovery.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_execution_time_recovery.setObjectName("label_text_execution_time_recovery")
        self.layout_execution_time_recovery.addWidget(self.label_text_execution_time_recovery)
        self.label_execution_time_recovery = QtWidgets.QLabel(self.widget_execution_time_recovery)
        self.label_execution_time_recovery.setText("")
        self.label_execution_time_recovery.setAlignment(QtCore.Qt.AlignCenter)
        self.label_execution_time_recovery.setObjectName("label_execution_time_recovery")
        self.layout_execution_time_recovery.addWidget(self.label_execution_time_recovery)
        self.verticalLayout_2.addWidget(self.widget_execution_time_recovery)
        self.widget_probability_error = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_probability_error.sizePolicy().hasHeightForWidth())
        self.widget_probability_error.setSizePolicy(sizePolicy)
        self.widget_probability_error.setObjectName("widget_probability_error")
        self.layout_probability_error = QtWidgets.QHBoxLayout(self.widget_probability_error)
        self.layout_probability_error.setContentsMargins(-1, 0, -1, -1)
        self.layout_probability_error.setObjectName("layout_probability_error")
        self.label_text_probability_error = QtWidgets.QLabel(self.widget_probability_error)
        self.label_text_probability_error.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_probability_error.setObjectName("label_text_probability_error")
        self.layout_probability_error.addWidget(self.label_text_probability_error)
        self.label_probability_error = QtWidgets.QLabel(self.widget_probability_error)
        self.label_probability_error.setText("")
        self.label_probability_error.setAlignment(QtCore.Qt.AlignCenter)
        self.label_probability_error.setObjectName("label_probability_error")
        self.layout_probability_error.addWidget(self.label_probability_error)
        self.verticalLayout_2.addWidget(self.widget_probability_error)
        self.pushButton_parameters = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_parameters.setCheckable(True)
        self.pushButton_parameters.setChecked(True)
        self.pushButton_parameters.setObjectName("pushButton_parameters")
        self.verticalLayout_2.addWidget(self.pushButton_parameters)
        self.groupBox_parameters = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_parameters.sizePolicy().hasHeightForWidth())
        self.groupBox_parameters.setSizePolicy(sizePolicy)
        self.groupBox_parameters.setObjectName("groupBox_parameters")
        self.layout_parameters = QtWidgets.QVBoxLayout(self.groupBox_parameters)
        self.layout_parameters.setContentsMargins(0, 5, 0, 9)
        self.layout_parameters.setSpacing(5)
        self.layout_parameters.setObjectName("layout_parameters")
        self.widget_sampling_rate = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_sampling_rate.sizePolicy().hasHeightForWidth())
        self.widget_sampling_rate.setSizePolicy(sizePolicy)
        self.widget_sampling_rate.setObjectName("widget_sampling_rate")
        self.layout_sampling_rate = QtWidgets.QHBoxLayout(self.widget_sampling_rate)
        self.layout_sampling_rate.setContentsMargins(-1, 0, -1, -1)
        self.layout_sampling_rate.setObjectName("layout_sampling_rate")
        self.label_text_sampling_rate = QtWidgets.QLabel(self.widget_sampling_rate)
        self.label_text_sampling_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_sampling_rate.setObjectName("label_text_sampling_rate")
        self.layout_sampling_rate.addWidget(self.label_text_sampling_rate)
        self.lineEdit_sampling_rate = QtWidgets.QLineEdit(self.widget_sampling_rate)
        self.lineEdit_sampling_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_sampling_rate.setObjectName("lineEdit_sampling_rate")
        self.layout_sampling_rate.addWidget(self.lineEdit_sampling_rate)
        self.layout_sampling_rate.setStretch(0, 1)
        self.layout_sampling_rate.setStretch(1, 1)
        self.layout_parameters.addWidget(self.widget_sampling_rate)
        self.widget_bit_counts = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_bit_counts.sizePolicy().hasHeightForWidth())
        self.widget_bit_counts.setSizePolicy(sizePolicy)
        self.widget_bit_counts.setObjectName("widget_bit_counts")
        self.layout_bit_counts = QtWidgets.QHBoxLayout(self.widget_bit_counts)
        self.layout_bit_counts.setContentsMargins(-1, 0, -1, -1)
        self.layout_bit_counts.setObjectName("layout_bit_counts")
        self.label_text_bit_counts = QtWidgets.QLabel(self.widget_bit_counts)
        self.label_text_bit_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_bit_counts.setObjectName("label_text_bit_counts")
        self.layout_bit_counts.addWidget(self.label_text_bit_counts)
        self.lineEdit_bit_counts = QtWidgets.QLineEdit(self.widget_bit_counts)
        self.lineEdit_bit_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_bit_counts.setObjectName("lineEdit_bit_counts")
        self.layout_bit_counts.addWidget(self.lineEdit_bit_counts)
        self.layout_bit_counts.setStretch(0, 1)
        self.layout_bit_counts.setStretch(1, 1)
        self.layout_parameters.addWidget(self.widget_bit_counts)
        self.widget_transfer_rate = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_transfer_rate.sizePolicy().hasHeightForWidth())
        self.widget_transfer_rate.setSizePolicy(sizePolicy)
        self.widget_transfer_rate.setObjectName("widget_transfer_rate")
        self.layout_transfer_rate = QtWidgets.QHBoxLayout(self.widget_transfer_rate)
        self.layout_transfer_rate.setContentsMargins(-1, 0, -1, -1)
        self.layout_transfer_rate.setObjectName("layout_transfer_rate")
        self.label_text_transfer_rate = QtWidgets.QLabel(self.widget_transfer_rate)
        self.label_text_transfer_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_transfer_rate.setObjectName("label_text_transfer_rate")
        self.layout_transfer_rate.addWidget(self.label_text_transfer_rate)
        self.lineEdit_transfer_rate = QtWidgets.QLineEdit(self.widget_transfer_rate)
        self.lineEdit_transfer_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_transfer_rate.setObjectName("lineEdit_transfer_rate")
        self.layout_transfer_rate.addWidget(self.lineEdit_transfer_rate)
        self.layout_parameters.addWidget(self.widget_transfer_rate)
        self.verticalLayout_2.addWidget(self.groupBox_parameters)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout_menu.addWidget(self.scrollArea)
        self.widget_number_experiments = QtWidgets.QWidget(self.widget_menu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_number_experiments.sizePolicy().hasHeightForWidth())
        self.widget_number_experiments.setSizePolicy(sizePolicy)
        self.widget_number_experiments.setObjectName("widget_number_experiments")
        self.layout_number_experiments = QtWidgets.QHBoxLayout(self.widget_number_experiments)
        self.layout_number_experiments.setContentsMargins(-1, 0, -1, -1)
        self.layout_number_experiments.setObjectName("layout_number_experiments")
        self.label_text_number_experiments = QtWidgets.QLabel(self.widget_number_experiments)
        self.label_text_number_experiments.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_number_experiments.setObjectName("label_text_number_experiments")
        self.layout_number_experiments.addWidget(self.label_text_number_experiments)
        self.lineEdit_number_experiments = QtWidgets.QLineEdit(self.widget_number_experiments)
        self.lineEdit_number_experiments.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_number_experiments.setObjectName("lineEdit_number_experiments")
        self.layout_number_experiments.addWidget(self.lineEdit_number_experiments)
        self.layout_number_experiments.setStretch(0, 1)
        self.layout_number_experiments.setStretch(1, 1)
        self.layout_menu.addWidget(self.widget_number_experiments)
        self.pushButton_probability_of_detection = QtWidgets.QPushButton(self.widget_menu)
        self.pushButton_probability_of_detection.setObjectName("pushButton_probability_of_detection")
        self.layout_menu.addWidget(self.pushButton_probability_of_detection)
        self.progressBar_probability = QtWidgets.QProgressBar(self.widget_menu)
        self.progressBar_probability.setProperty("value", 0)
        self.progressBar_probability.setObjectName("progressBar_probability")
        self.layout_menu.addWidget(self.progressBar_probability)
        self.widget_execution_time = QtWidgets.QWidget(self.widget_menu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_execution_time.sizePolicy().hasHeightForWidth())
        self.widget_execution_time.setSizePolicy(sizePolicy)
        self.widget_execution_time.setObjectName("widget_execution_time")
        self.layout_execution_time = QtWidgets.QHBoxLayout(self.widget_execution_time)
        self.layout_execution_time.setContentsMargins(-1, 0, -1, -1)
        self.layout_execution_time.setObjectName("layout_execution_time")
        self.label_text_execution_time = QtWidgets.QLabel(self.widget_execution_time)
        self.label_text_execution_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_execution_time.setObjectName("label_text_execution_time")
        self.layout_execution_time.addWidget(self.label_text_execution_time)
        self.label_execution_time = QtWidgets.QLabel(self.widget_execution_time)
        self.label_execution_time.setText("")
        self.label_execution_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_execution_time.setObjectName("label_execution_time")
        self.layout_execution_time.addWidget(self.label_execution_time)
        self.layout_menu.addWidget(self.widget_execution_time)
        self.horizontalLayout.addWidget(self.widget_menu)
        self.widget_main_1 = QtWidgets.QWidget(Dialog)
        self.widget_main_1.setObjectName("widget_main_1")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.widget_main_1)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.widget_plot = QtWidgets.QWidget(self.widget_main_1)
        self.widget_plot.setObjectName("widget_plot")
        self.layout_plot = QtWidgets.QVBoxLayout(self.widget_plot)
        self.layout_plot.setObjectName("layout_plot")
        self.verticalLayout_1.addWidget(self.widget_plot)
        self.widget_plot_3 = QtWidgets.QWidget(self.widget_main_1)
        self.widget_plot_3.setObjectName("widget_plot_3")
        self.layout_plot_3 = QtWidgets.QVBoxLayout(self.widget_plot_3)
        self.layout_plot_3.setObjectName("layout_plot_3")
        self.verticalLayout_1.addWidget(self.widget_plot_3)
        self.horizontalLayout.addWidget(self.widget_main_1)
        self.widget_main = QtWidgets.QWidget(Dialog)
        self.widget_main.setObjectName("widget_main")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_main)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_plot_2 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_2.setObjectName("widget_plot_2")
        self.layout_plot_2 = QtWidgets.QVBoxLayout(self.widget_plot_2)
        self.layout_plot_2.setObjectName("layout_plot_2")
        self.verticalLayout.addWidget(self.widget_plot_2)
        self.widget_plot_4 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_4.setObjectName("widget_plot_4")
        self.layout_plot_4 = QtWidgets.QVBoxLayout(self.widget_plot_4)
        self.layout_plot_4.setObjectName("layout_plot_4")
        self.verticalLayout.addWidget(self.widget_plot_4)
        self.horizontalLayout.addWidget(self.widget_main)

        self.retranslateUi(Dialog)
        self.pushButton_signal_generation.clicked['bool'].connect(self.widget_loading_data.setVisible) # type: ignore
        self.pushButton_parameters.clicked['bool'].connect(self.groupBox_parameters.setVisible) # type: ignore
        self.radioButton_own.toggled['bool'].connect(self.lineEdit_bits.setVisible) # type: ignore
        self.radioButton_random.toggled['bool'].connect(self.lineEdit_bits.setEnabled) # type: ignore
        self.radioButton_own.toggled['bool'].connect(self.lineEdit_bits.setEnabled) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_signal_generation.setText(_translate("Dialog", "Моделирование сигнала"))
        self.groupBox.setTitle(_translate("Dialog", "Выбор последовательности"))
        self.radioButton_random.setText(_translate("Dialog", "Случайная"))
        self.radioButton_own.setText(_translate("Dialog", "Своя"))
        self.lineEdit_bits.setText(_translate("Dialog", "101101"))
        self.groupBox_envelope_parameters.setTitle(_translate("Dialog", "Параметры для огибающей"))
        self.label_text_carrier_frequency.setText(_translate("Dialog", "Несущая частота, Гц"))
        self.lineEdit_carrier_frequency.setText(_translate("Dialog", "1000"))
        self.label_text_phase.setText(_translate("Dialog", "Фаза, рад"))
        self.lineEdit_phase.setText(_translate("Dialog", "0"))
        self.pushButton_simulate_signal.setText(_translate("Dialog", "Смоделировать сигнал"))
        self.groupBox_noise.setTitle(_translate("Dialog", "Шум для сигнала"))
        self.label_text_noise.setText(_translate("Dialog", "дБ"))
        self.lineEdit_noise.setText(_translate("Dialog", "0"))
        self.pushButton_add_noise.setText(_translate("Dialog", "Добавить шум"))
        self.pushButton_restore_the_original_information.setText(_translate("Dialog", "Восстановить исходную информацию"))
        self.label_text_execution_time_recovery.setText(_translate("Dialog", "Время выполнения:"))
        self.label_text_probability_error.setText(_translate("Dialog", "Вероятность ошибки"))
        self.pushButton_parameters.setText(_translate("Dialog", "Параметры"))
        self.groupBox_parameters.setTitle(_translate("Dialog", "Параметры "))
        self.label_text_sampling_rate.setText(_translate("Dialog", "Частота дискр., Гц"))
        self.lineEdit_sampling_rate.setText(_translate("Dialog", "10000"))
        self.label_text_bit_counts.setText(_translate("Dialog", "Число бит"))
        self.lineEdit_bit_counts.setText(_translate("Dialog", "10"))
        self.label_text_transfer_rate.setText(_translate("Dialog", "V перед. данных, бит/с"))
        self.lineEdit_transfer_rate.setText(_translate("Dialog", "1000"))
        self.label_text_number_experiments.setText(_translate("Dialog", "Количество экспериментов"))
        self.lineEdit_number_experiments.setText(_translate("Dialog", "100"))
        self.pushButton_probability_of_detection.setText(_translate("Dialog", "Провести исследование"))
        self.label_text_execution_time.setText(_translate("Dialog", "Время выполнения:"))