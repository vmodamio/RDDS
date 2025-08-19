#!/usr/bin/env python3
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Signal, Qt
import time
import numpy as np
import os
from datetime import datetime, timedelta
from rdds_fit_single import rdds_fit_single, decay

class rdds_app(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Decay curves")
        appname  = QtWidgets.QLabel("Rancour-free RDDS analyzer")
        appname.setObjectName('title')

        self.rdds = rdds_fit_single()
        self.fileR = "zr100_4plus.dat"
        self.fileF = "zr100_6plus.dat"

        self.nodes = None
        self.maxKnots = 10

        self.mw = QtWidgets.QWidget()
        self.setCentralWidget(self.mw)
        layout_main = QtWidgets.QHBoxLayout()
        self.mw.setLayout(layout_main)
        self.setGeometry(100,100,1200, 900)

        left_column = QtWidgets.QFrame()
        left_column.setObjectName('left')
        left_column.setMinimumWidth(380)
        left_column.setMaximumWidth(540)
        layout_left_column = QtWidgets.QVBoxLayout()
        left_column.setLayout(layout_left_column)
        
        decay_settings = QtWidgets.QWidget()
        layout_decay_settings = QtWidgets.QGridLayout()
        decay_settings.setLayout(layout_decay_settings)
        ldecay  = QtWidgets.QLabel("Main decay (R): ")
        lfeeder = QtWidgets.QLabel("Feeder decay (F): ")
        lfratio = QtWidgets.QLabel("Feeding factor")
        lfratio_error = QtWidgets.QLabel("err")

        fit_settings = QtWidgets.QWidget()
        layout_fit_settings = QtWidgets.QGridLayout()
        fit_settings.setLayout(layout_fit_settings)

        self.analysis_type = QtWidgets.QComboBox()
        self.analysis_type.addItems(['Single-\u03B3 with feeding decay','Coincidence-\u03B3 with prompt feeder'])

        self.file_decay = ClickableLineEdit(0)
        self.file_decay.clicked.connect(self.change_browse)
        self.file_decay.editingFinished.connect(self.setDecay)
        self.file_decay.setPlaceholderText("Select file...")

        self.file_feeder = ClickableLineEdit(1)
        self.file_feeder.clicked.connect(self.change_browse)
        self.file_feeder.editingFinished.connect(self.setFeeder)
        self.file_feeder.setPlaceholderText("Select file...")

        self.feeding_ratio = QtWidgets.QLineEdit()
        self.feeding_ratio.editingFinished.connect(self.setFratio)
        self.feeding_error = QtWidgets.QLineEdit()
        self.frac_validator = QtGui.QDoubleValidator(0.0, 1.0, 5)
        self.frac_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.feeding_ratio.setValidator(self.frac_validator)
        self.feeding_error.setValidator(self.frac_validator)
        self.feeding_ratio.setText("0.75")
        self.feeding_error.setText("0.05")


        self.browse_decay = QtWidgets.QPushButton("Browse")
        self.browse_decay.clicked.connect(self.open_file_dialog)

        self.splineorder = QtWidgets.QSpinBox()
        self.splineorder.setRange(2,8)
        self.splineorder.setValue(3)
        self.knots = QtWidgets.QSpinBox()
        self.knots.setRange(2,self.maxKnots)
        self.knots.setValue(3)
        self.numdatapoints = 0
        self.sbase = 0
        self.sknots = 0
        self.ndf = 0
        
        self.tauguess = QtWidgets.QLineEdit()
        self.tauguess.editingFinished.connect(self.setGuess)
        self.tau_validator = QtGui.QDoubleValidator(0.1, 100, 1)
        self.tau_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.tauguess.setValidator(self.tau_validator)
        self.ftol = QtWidgets.QLineEdit()
        self.ftol.editingFinished.connect(self.setTolerance)
        self.ftol_validator = QtGui.QDoubleValidator(1e-12, 1e-4, 1)
        self.ftol_validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self.ftol.setValidator(self.ftol_validator)
        self.tauguess.setText("5.0")
        self.ftol.setText("1e-8")

        self.fit_lifetime = QtWidgets.QLineEdit()
        self.fit_lifetime.setObjectName('readout')
        self.fit_lifetime.setReadOnly(True)
        self.fit_lifetime.setAlignment(Qt.AlignRight)
        self.fit_lifetime.setText('None')

        self.fit_result = QtWidgets.QLineEdit()
        self.fit_result.setObjectName('readout')
        self.fit_result.setReadOnly(True)
        self.fit_result.setText('None')

        self.fit_ndf = QtWidgets.QLineEdit()
        self.fit_ndf.setAlignment(Qt.AlignRight)
        self.fit_ndf.setObjectName('readout')
        self.fit_ndf.setReadOnly(True)
        self.fit_ndf.setText(f'{self.numdatapoints - 2*self.sbase}')

        self.data_points = QtWidgets.QLineEdit()
        self.data_points.setObjectName('readout')
        self.data_points.setAlignment(Qt.AlignRight)
        self.data_points.setReadOnly(True)
        self.data_points.setText(f'{self.numdatapoints}')

        self.spline_base = QtWidgets.QLineEdit()
        self.spline_base.setObjectName('readout')
        self.spline_base.setAlignment(Qt.AlignRight)
        self.spline_base.setReadOnly(True)
        self.spline_base.setText(f'{self.sbase}')
        self.knots.valueChanged.connect(self.setSpline)
        self.splineorder.valueChanged.connect(self.setSpline)
        self.sbase = 0

        self.fit_eff_ndf = QtWidgets.QLineEdit()
        self.fit_eff_ndf.setObjectName('readout')
        self.fit_eff_ndf.setReadOnly(True)
        self.fit_eff_ndf.setText('None')

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.runFit)

        layout_decay_settings.addWidget(QtWidgets.QLabel("Analysis"),           0, 0)
        layout_decay_settings.addWidget(self.analysis_type,  0, 1, 1, 5)
        layout_decay_settings.addWidget(ldecay,           1, 0)
        layout_decay_settings.addWidget(self.file_decay,  1, 1, 1, 5)
        layout_decay_settings.addWidget(lfeeder,          2, 0)
        layout_decay_settings.addWidget(self.file_feeder, 2, 1, 1, 5)
        layout_decay_settings.addWidget(lfratio,          3, 0)
        layout_decay_settings.addWidget(lfratio_error,    3, 2)
        layout_decay_settings.addWidget(self.feeding_ratio   , 3, 1)
        layout_decay_settings.addWidget(self.feeding_error   , 3, 3)
        layout_decay_settings.addWidget(QtWidgets.QLabel("Spline order")   , 4, 0)
        layout_decay_settings.addWidget(self.splineorder   , 4, 1)
        layout_decay_settings.addWidget(QtWidgets.QLabel("Num knots")   , 4, 2)
        layout_decay_settings.addWidget(self.knots   , 4, 3)
        layout_decay_settings.addWidget(QtWidgets.QLabel("Base")   , 4, 4)
        layout_decay_settings.addWidget(self.spline_base   , 4, 5)
        layout_decay_settings.addWidget(QtWidgets.QLabel("Sum pts")   , 3, 4)
        layout_decay_settings.addWidget(self.data_points   , 3, 5)
        layout_decay_settings.addWidget(QtWidgets.QLabel("ndf")   , 5, 4)
        layout_decay_settings.addWidget(self.fit_ndf   , 5, 5)

        layout_fit_settings.addWidget(QtWidgets.QLabel("Tau guess (ps)"),           0, 0)
        layout_fit_settings.addWidget(self.tauguess,  0, 1)

        layout_fit_settings.setColumnStretch(3, 3)
        layout_fit_settings.addWidget(QtWidgets.QLabel("ftol"),           1, 0)
        layout_fit_settings.addWidget(self.ftol,  1, 1)
        layout_fit_settings.addWidget(QtWidgets.QLabel("Lifetime"),           2, 0)
        layout_fit_settings.addWidget(self.fit_lifetime,  2, 1 )
        layout_fit_settings.addWidget(self.run_button,  2, 2)
        layout_fit_settings.addWidget(QtWidgets.QLabel("Fit result"),           3, 0)
        layout_fit_settings.addWidget(self.fit_result,  3, 1, 1,3)

        layout_main.addWidget(left_column, stretch=0, alignment=Qt.AlignLeft)

        main_column = QtWidgets.QFrame()
        layout_body = QtWidgets.QVBoxLayout()
        main_column.setLayout(layout_body)

        layout_body.addWidget(main_column)
        layout_main.addWidget(main_column, stretch=1, alignment=Qt.AlignRight)


        self.quitbutton = QtWidgets.QPushButton("Quit")
        self.quitbutton.clicked.connect(self.close)
        self.export = QtWidgets.QPushButton("Export Figure")
        self.export.clicked.connect(self.save_figure)

        #layout_left_column.addLayout(self.flo)
        upper_buttons = QtWidgets.QWidget()
        upper_layout = QtWidgets.QHBoxLayout()
        upper_buttons.setLayout(upper_layout)
        upper_layout.addWidget(self.quitbutton)
        upper_layout.addWidget(self.browse_decay)
        upper_layout.addWidget(self.export)
        layout_left_column.addWidget(appname)
        layout_left_column.addWidget(upper_buttons)
        layout_left_column.addWidget(decay_settings)
        layout_left_column.addStretch()

        self.browse = 0

        layout_body.addWidget(self.rdds.canvas)
        layout_body.setContentsMargins(0,0,0,0)
        layout_body.setSpacing(0)

        R = decay("zr100_4plus.dat", self.rdds.spline_order, self.rdds.mesh, 1.0, 0.0)
        F = decay("zr100_6plus.dat", self.rdds.spline_order, self.rdds.mesh, 0.83, 0.05)
        self.rdds.setDecay(R)
        self.rdds.setFeeder(F)

        self.node_settings = QtWidgets.QWidget()
        self.node_settings_layout = QtWidgets.QHBoxLayout()
        self.node_buttons_layout = QtWidgets.QVBoxLayout()
        self.node_settings.setLayout(self.node_settings_layout)
        self.node_layout = QtWidgets.QFormLayout()
        self.nodes = [QtWidgets.QDoubleSpinBox() for _ in range(self.maxKnots)]
        #self.nodes_lineEdit = [node.lineEdit() for node in self.nodes]
        self.nodes[0].setReadOnly(True)
        self.nodes[0].setObjectName('readonly')
        for i in range(self.maxKnots):
            self.node_layout.addRow(f'k{i}', self.nodes[i])
            self.nodes[i].setMaximumWidth(100)
            self.nodes[i].lineEdit().setAlignment(Qt.AlignRight)
            #self.nodes[i].lineEdit().editingFinished.connect(self.changeNode)
            self.nodes[i].valueChanged.connect(self.changeNode)
        #layout_left_column.addLayout(self.node_layout)
        self.button_arrangeLog = QtWidgets.QPushButton("Arrange Log from K1")
        self.button_arrangeLog.clicked.connect(self.setLog)
        self.button_arrangeLin = QtWidgets.QPushButton("Arrange Lin til Kn")
        self.button_arrangeLin.clicked.connect(self.setLin)
        self.node_buttons_layout.addWidget(self.button_arrangeLog)
        self.node_buttons_layout.addWidget(self.button_arrangeLin)
        self.node_settings_layout.addLayout(self.node_buttons_layout)
        self.node_settings_layout.addLayout(self.node_layout)

        layout_left_column.addWidget(self.node_settings)
        layout_left_column.addWidget(fit_settings)
        self.setNodes()

        self.setNumDataPoints()
        self.rdds.runFit()
        self.rdds.drawFigure()


    def runFit(self):
        #setDecay()
        #setFeeder()
        self.rdds.runFit()
        self.rdds.drawFigure()
        #self.fit_lifetime.setText(f'{self.rdds.tau_result} \u00B1 {self.rdds.tau_error} ps')
        self.fit_lifetime.setText('{0:6.2f} \u00B1 {1:4.2f}  ps'.format(self.rdds.tau_result,self.rdds.tau_error))
        time = '{0:10.3f} ms'.format(self.rdds.fit_time)
        chi2 = '{0:8.5f}'.format(0.5*self.rdds.minout.fun)
        success = 'Success' if self.rdds.minout.success else 'Fail'
        self.fit_result.setText(f'{success},{time},  {self.rdds.minout.nit} Iterations, \u03C7\u00B2 = {chi2}')

    def setLog(self):
        self.rdds.setMeshMode('log')
        self.setSpline(None)
        self.setNodes()
        #self.rdds.figure.canvas.draw()
        self.rdds.clearMesh()
        self.rdds.drawMesh()
        self.rdds.canvas.draw()
    def setLin(self):
        self.rdds.setMeshMode('lin')
        self.setSpline(None)
        self.setNodes()
        self.rdds.clearMesh()
        self.rdds.drawMesh()
        self.rdds.canvas.draw()

    def changeNode(self, value):
        emmiter = self.sender()
        idx = self.nodes.index(emmiter)
        #value = float(self.nodes_lineEdit[idx].text())
        print(f'Node [{idx}] changed to value: {value} ')
        self.rdds.mesh[idx] = value
        if idx == 1:
            self.rdds.mesh_log_first = value
        elif idx == (self.knots.value() -1):
            self.rdds.upperL = value
        self.setDecay(self.fileR)
        self.setFeeder(self.fileF)
        self.rdds.clearMesh()
        self.rdds.drawMesh()

    def setNodes(self):
        for i in range(self.maxKnots):
            if i < self.knots.value():
                self.nodes[i].setValue(self.rdds.mesh[i])
                #self.nodes[i].show()
                self.node_layout.setRowVisible(self.nodes[i], True)
            else:
                #self.nodes[i].hide()
                self.node_layout.setRowVisible(self.nodes[i], False)

    def setDecay(self, text):
        R = decay(self.fileR, self.rdds.spline_order, self.rdds.mesh, 1.0, 0.0)
        self.rdds.setDecay(R)
        self.setNumDataPoints()
    def setFeeder(self, text):
        F = decay(self.fileF, self.rdds.spline_order, self.rdds.mesh, 
                  float(self.feeding_ratio.text()), float(self.feeding_error.text()))
        self.rdds.setFeeder(F)
        self.setNumDataPoints()
    def setFratio(self):
        self.rdds.F.fratio = float(self.feeding_ratio.text())
    def setGuess(self):
        self.rdds.tauguess = float(self.tauguess.text())
    def setTolerance(self):
        self.rdds.ftol = float(self.ftol.text())

    def setSpline(self, either):
        base = self.knots.value() + self.splineorder.value() -2
        self.spline_base.setText(f'{base}')
        self.sbase = base
        self.fit_ndf.setText(f'{self.numdatapoints - 2*self.sbase -1}')
        self.rdds.spline_order = self.splineorder.value()
        #self.rdds.mesh_log_first = 
        #self.rdds.upperL = 
        self.rdds.spline_knots = self.knots.value()
        self.rdds.setMesh()
        self.setDecay(self.fileR)
        self.setFeeder(self.fileF)
        self.setNodes()
        return base

    def setNumDataPoints(self):
        if isinstance(self.rdds.R, decay) and isinstance(self.rdds.F, decay):
            self.numdatapoints =  len(self.rdds.R.x)+len(self.rdds.F.x)
        else:
            self.numdatapoints = 0
        self.fit_ndf.setText(f'{self.numdatapoints - 2*self.sbase -1}')
        self.data_points.setText(f'{self.numdatapoints}')
        self.spline_base.setText(f'{self.sbase}')

    def change_browse(self, decay):
        self.browse = decay
        print(self.browse)

    def open_file_dialog(self):
        # Open a file dialog to select a single file
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a File",
            dir=".",  # Default directory
            filter="All Files (*.*);;Text Files (*.txt);;Python Files (*.py)" # File filters
        )
        if filename: # Check if a file was selected (not cancelled)
            self.rdds.spline_order = self.splineorder.value()
            basename = os.path.basename(filename)
            if (self.browse == 0):
                self.file_decay.setText(basename)
                self.fileR = filename
                self.setDecay(filename)
                #R = decay(filename, self.rdds.spline_order, self.rdds.mesh, 1.0)
                #self.rdds.setDecay(filename)
            else:
                self.file_feeder.setText(basename)
                self.fileF = filename
                self.setFeeder(filename)
                #F = decay(filename, self.rdds.spline_order, self.rdds.mesh, 0.83)
                #self.rdds.setFeeder(filename)

    def save_figure(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)")
        if file_path:
            self.rdds.canvas.figure.savefig(file_path)
            print(f"Figure saved to: {file_path}")



class ClickableLineEdit(QtWidgets.QLineEdit):
    clicked = Signal(int) # Define a custom signal
    def __init__(self, decay, parent=None):
        super().__init__(parent)
        self.decay = decay
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.decay) # Emit the custom signal on left-click
        super().mousePressEvent(event) # Call the base class implementation

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont("Resources/SourceSansPro/SourceSansPro-Regular.ttf")
    QtGui.QFontDatabase.addApplicationFont("Resources/SourceSansPro/SourceSansPro-Bold.ttf")
    QtGui.QFontDatabase.addApplicationFont("Resources/NotoSansMono/NotoSansMono-VariableFont_wdth,wght.ttf")
    myqss = open('Resources/gui.qss').read()
    app.setStyleSheet(myqss)
    myapp = rdds_app()
    myapp.show()
    sys.exit(app.exec())

