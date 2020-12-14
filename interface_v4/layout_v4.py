from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
import os
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pylab import get_cmap
import scipy
import seaborn as sb
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs

os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-3.6.2'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\R\\R-3.6.2\\bin\\x64\\'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\R\\R-3.6.2\\'

import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

base = rpackages.importr("base")
utils = rpackages.importr("utils")

from to_plot import *

class DataFrameModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1072, 599)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Stacked widget for plot configs
        self.PlotOptionsStackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.PlotOptionsStackedWidget.setGeometry(QtCore.QRect(0, 40, 191, 391))
        self.PlotOptionsStackedWidget.setObjectName("PlotOptionsStackedWidget")
        ## Barnes Hut page (0)
        self.BarnesHut_page = QtWidgets.QWidget()
        self.BarnesHut_page.setObjectName("BarnesHut_page")
        self.label_13 = QtWidgets.QLabel(self.BarnesHut_page)
        self.label_13.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.label_13.setObjectName("label_13")
        self.label_23 = QtWidgets.QLabel(self.BarnesHut_page)
        self.label_23.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.label_23.setObjectName("label_12")
        self.label_6 = QtWidgets.QLabel(self.BarnesHut_page)
        self.label_6.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.BarnesHut_page)
        self.label_7.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.label_7.setObjectName("label_7")
        self.label_14 = QtWidgets.QLabel(self.BarnesHut_page)
        self.label_14.setGeometry(QtCore.QRect(10, 300, 161, 16))
        self.label_14.setObjectName("label_14")
        self.BarnesHut_Damping = QtWidgets.QDoubleSpinBox(self.BarnesHut_page)
        self.BarnesHut_Damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.BarnesHut_Damping.setObjectName("BarnesHut_Damping")
        self.BarnesHut_SpringLength = QtWidgets.QSpinBox(self.BarnesHut_page)
        self.BarnesHut_SpringLength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.BarnesHut_SpringLength.setObjectName("BarnesHut_SpringLength")
        self.BarnesHut_CentralGravity = QtWidgets.QDoubleSpinBox(self.BarnesHut_page)
        self.BarnesHut_CentralGravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.BarnesHut_CentralGravity.setObjectName("BarnesHut_CentralGravity")
        self.BarnesHut_GravityCtt = QtWidgets.QSpinBox(self.BarnesHut_page)
        self.BarnesHut_GravityCtt.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.BarnesHut_GravityCtt.setObjectName("BarnesHut_GravityCtt")
        self.BarnesHut_SpringCtt = QtWidgets.QDoubleSpinBox(self.BarnesHut_page)
        self.BarnesHut_SpringCtt.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.BarnesHut_SpringCtt.setObjectName("BarnesHut_SpringCtt")
        self.BarnesHutOverlap = QtWidgets.QCheckBox(self.BarnesHut_page)
        self.BarnesHutOverlap.setGeometry(QtCore.QRect(10, 360, 161, 17))
        self.BarnesHutOverlap.setObjectName("BarnesHutOverlap")
        self.PlotOptionsStackedWidget.addWidget(self.BarnesHut_page)
        self.BarnesHut_GravityCtt.setRange(-99999, 0)
        self.BarnesHut_GravityCtt.setSingleStep(10)
        self.BarnesHut_CentralGravity.setRange(0, 1)
        self.BarnesHut_CentralGravity.setSingleStep(0.1)
        self.BarnesHut_SpringLength.setRange(0, 999)
        self.BarnesHut_SpringLength.setSingleStep(5)
        self.BarnesHut_SpringCtt.setRange(0, 1)
        self.BarnesHut_SpringCtt.setSingleStep(0.01)
        self.BarnesHut_Damping.setRange(0, 1)
        self.BarnesHut_Damping.setSingleStep(0.01)
        self.BarnesHut_GravityCtt.setValue(-2000)
        self.BarnesHut_CentralGravity.setValue(0.3)
        self.BarnesHut_SpringLength.setValue(95)
        self.BarnesHut_SpringCtt.setValue(0.04)
        self.BarnesHut_Damping.setValue(0.09)
        self.BarnesHut_CentralGravity.setDecimals(2)
        self.BarnesHut_SpringCtt.setDecimals(2)
        self.BarnesHut_Damping.setDecimals(2)
        ## Hierarchical page (1)
        self.hierarchical_page = QtWidgets.QWidget()
        self.hierarchical_page.setObjectName("hierarchical_page")
        self.label_2 = QtWidgets.QLabel(self.hierarchical_page)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.hierarchical_page)
        self.label_3.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.hierarchical_page)
        self.label_4.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.hierarchical_page)
        self.label_5.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.label_5.setObjectName("label_5")
        self.label_15 = QtWidgets.QLabel(self.hierarchical_page)
        self.label_15.setGeometry(QtCore.QRect(10, 300, 161, 16))
        self.label_15.setObjectName("label_15")
        self.hierarchical_node_distance = QtWidgets.QSpinBox(self.hierarchical_page)
        self.hierarchical_node_distance.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.hierarchical_node_distance.setObjectName("hierarchical_node_distance")
        self.hierarchical_gravity = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_gravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.hierarchical_gravity.setObjectName("hierarchical_gravity")
        self.hierarchical_springlength = QtWidgets.QSpinBox(self.hierarchical_page)
        self.hierarchical_springlength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.hierarchical_springlength.setObjectName("hierarchical_springlength")
        self.hierarchical_springconstant = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_springconstant.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.hierarchical_springconstant.setObjectName("hierarchical_springconstant")
        self.hierarchical_damping = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.hierarchical_damping.setObjectName("hierarchical_damping")
        self.PlotOptionsStackedWidget.addWidget(self.hierarchical_page)
        self.hierarchical_node_distance.setRange(0, 9999)
        self.hierarchical_node_distance.setSingleStep(5)
        self.hierarchical_gravity.setRange(0, 1)
        self.hierarchical_gravity.setSingleStep(0.01)
        self.hierarchical_springlength.setRange(0, 999)
        self.hierarchical_springlength.setSingleStep(1)
        self.hierarchical_springconstant.setRange(0, 1)
        self.hierarchical_springconstant.setSingleStep(0.01)
        self.hierarchical_damping.setRange(0, 1)
        self.hierarchical_damping.setSingleStep(0.01)
        self.hierarchical_node_distance.setValue(120)
        self.hierarchical_gravity.setValue(0.00)
        self.hierarchical_springlength.setValue(100)
        self.hierarchical_springconstant.setValue(0.01)
        self.hierarchical_damping.setValue(0.09)
        self.hierarchical_gravity.setDecimals(2)
        self.hierarchical_springconstant.setDecimals(2)
        self.hierarchical_damping.setDecimals(2)
        ## Repulsion page (2)
        self.Repulsion_page = QtWidgets.QWidget()
        self.Repulsion_page.setObjectName("Repulsion_page")
        self.label_11 = QtWidgets.QLabel(self.Repulsion_page)
        self.label_11.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.label_11.setObjectName("label_11")
        self.label_8 = QtWidgets.QLabel(self.Repulsion_page)
        self.label_8.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.Repulsion_page)
        self.label_9.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.Repulsion_page)
        self.label_10.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.Repulsion_page)
        self.label_12.setGeometry(QtCore.QRect(10, 300, 47, 13))
        self.label_12.setObjectName("label_12")
        self.repulsion_node_distance = QtWidgets.QSpinBox(self.Repulsion_page)
        self.repulsion_node_distance.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.repulsion_node_distance.setObjectName("repulsion_node_distance")
        self.repulsion_springconstant = QtWidgets.QDoubleSpinBox(self.Repulsion_page)
        self.repulsion_springconstant.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.repulsion_springconstant.setObjectName("repulsion_springconstant")
        self.repulsion_springlength = QtWidgets.QSpinBox(self.Repulsion_page)
        self.repulsion_springlength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.repulsion_springlength.setObjectName("repulsion_springlength")
        self.repulsion_gravity = QtWidgets.QDoubleSpinBox(self.Repulsion_page)
        self.repulsion_gravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.repulsion_gravity.setObjectName("repulsion_gravity")
        self.repulsion_damping = QtWidgets.QDoubleSpinBox(self.Repulsion_page)
        self.repulsion_damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.repulsion_damping.setObjectName("repulsion_damping")
        self.PlotOptionsStackedWidget.addWidget(self.Repulsion_page)
        self.repulsion_node_distance.setRange(0, 9999)
        self.repulsion_node_distance.setSingleStep(5)
        self.repulsion_gravity.setRange(0, 1)
        self.repulsion_gravity.setSingleStep(0.05)
        self.repulsion_springlength.setRange(0, 999)
        self.repulsion_springlength.setSingleStep(5)
        self.repulsion_springconstant.setRange(0, 1)
        self.repulsion_springconstant.setSingleStep(0.01)
        self.repulsion_damping.setRange(0, 1)
        self.repulsion_damping.setSingleStep(0.01)
        self.repulsion_node_distance.setValue(100)
        self.repulsion_gravity.setValue(0.2)
        self.repulsion_springlength.setValue(200)
        self.repulsion_springconstant.setValue(0.05)
        self.repulsion_damping.setValue(0.09)
        self.repulsion_gravity.setDecimals(2)
        self.repulsion_springconstant.setDecimals(2)
        ## Forced Atlas (3)
        self.ForcedAtlas_page = QtWidgets.QWidget()
        self.ForcedAtlas_page.setObjectName("ForcedAtlas_page")
        self.ForcedAtlas_Damping = QtWidgets.QDoubleSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_Damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.ForcedAtlas_Damping.setObjectName("ForcedAtlas_Damping")
        self.label_16 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.label_16.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.label_17.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.label_18.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.label_19.setGeometry(QtCore.QRect(10, 300, 47, 13))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.label_20.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.label_20.setObjectName("label_20")
        self.ForcedAtlas_SpringLength = QtWidgets.QSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_SpringLength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.ForcedAtlas_SpringLength.setObjectName("ForcedAtlas_SpringLength")
        self.ForcedAtlas_CentralGravity = QtWidgets.QDoubleSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_CentralGravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.ForcedAtlas_CentralGravity.setObjectName("ForcedAtlas_CentralGravity")
        self.ForcedAtlas_GravityCtt = QtWidgets.QSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_GravityCtt.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.ForcedAtlas_GravityCtt.setObjectName("ForcedAtlas_GravityCtt")
        self.ForcedAtlas_SpringCtt = QtWidgets.QDoubleSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_SpringCtt.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.ForcedAtlas_SpringCtt.setObjectName("ForcedAtlas_SpringCtt")
        self.ForcedAtlasOverlap = QtWidgets.QCheckBox(self.ForcedAtlas_page)
        self.ForcedAtlasOverlap.setGeometry(QtCore.QRect(10, 360, 161, 17))
        self.ForcedAtlasOverlap.setObjectName("ForcedAtlasOverlap")
        self.PlotOptionsStackedWidget.addWidget(self.ForcedAtlas_page)
        self.ForcedAtlas_GravityCtt.setRange(-99999, 0)
        self.ForcedAtlas_GravityCtt.setSingleStep(5)
        self.ForcedAtlas_CentralGravity.setRange(0, 1)
        self.ForcedAtlas_CentralGravity.setSingleStep(0.01)
        self.ForcedAtlas_SpringLength.setRange(0, 999)
        self.ForcedAtlas_SpringLength.setSingleStep(5)
        self.ForcedAtlas_SpringCtt.setRange(0, 1)
        self.ForcedAtlas_SpringCtt.setSingleStep(0.01)
        self.ForcedAtlas_Damping.setRange(0, 1)
        self.ForcedAtlas_Damping.setSingleStep(0.1)
        self.ForcedAtlas_GravityCtt.setValue(-50)
        self.ForcedAtlas_CentralGravity.setValue(0.01)
        self.ForcedAtlas_SpringLength.setValue(100)
        self.ForcedAtlas_SpringCtt.setValue(0.08)
        self.ForcedAtlas_Damping.setValue(0.4)
        self.ForcedAtlas_CentralGravity.setDecimals(2)
        self.ForcedAtlas_SpringCtt.setDecimals(2)
        self.ForcedAtlas_Damping.setDecimals(2)
        ## No config page (4)
        self.NoConfig_page = QtWidgets.QWidget()
        self.NoConfig_page.setObjectName("NoConfig_page")
        self.PlotOptionsStackedWidget.addWidget(self.NoConfig_page)

        # ComboBox to select the plots
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(10, 0, 161, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.currentIndexChanged.connect(self.getGraphType) #Change the graph tab ONLY if the ComboBox index changed

        # Timestep slider
        self.timestep_slider = QtWidgets.QSlider(self.centralwidget)
        self.timestep_slider.setGeometry(QtCore.QRect(10, 520, 161, 31))
        self.timestep_slider.setOrientation(QtCore.Qt.Horizontal)
        self.timestep_slider.setObjectName("timestep_slider")
        self.label = QtWidgets.QLabel(self.centralwidget)  # Timestep label
        self.label.setGeometry(QtCore.QRect(6, 494, 161, 20))
        self.label.setObjectName("label")
        self.timestep_slider.valueChanged.connect(self.updateSliderLabel) # Update the slider label

        # Tab browser widget
        self.TabBrowserWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.TabBrowserWidget.setGeometry(QtCore.QRect(190, 0, 691, 551))
        self.TabBrowserWidget.setObjectName("TabBrowserWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        ##  Plot widget
        self.PlotWidget = QtWebEngineWidgets.QWebEngineView(self.tab)
        self.PlotWidget.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.PlotWidget.setObjectName("PlotWidget")
        self.TabBrowserWidget.addTab(self.tab, "")
        ## Raw data view widget
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableView = QtWidgets.QTableView(self.tab_2)
        self.tableView.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.tableView.setObjectName("tableView")
        self.TabBrowserWidget.addTab(self.tab_2, "")
        ## Nodes view widget
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_2")
        self.tableView_2 = QtWidgets.QTableView(self.tab_3)
        self.tableView_2.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.tableView_2.setObjectName("tableView_2")
        self.TabBrowserWidget.addTab(self.tab_3, "")

        # Apply changes button
        self.ApplyChanges = QtWidgets.QPushButton(self.centralwidget)
        self.ApplyChanges.setGeometry(QtCore.QRect(14, 452, 161, 31))
        self.ApplyChanges.setObjectName("ApplyChanges")
        MainWindow.setCentralWidget(self.centralwidget)
        self.ApplyChanges.setEnabled(False) # The button is just enabled when the user select a file
        self.ApplyChanges.clicked.connect(self.BtnAction) # If the button is pressed, then go to the action

        # Stacked widgets for statistics
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(880, 20, 191, 531))
        self.stackedWidget.setObjectName("stackedWidget")
        ## Page for standard plots (graphs)
        self.StandardPlots = QtWidgets.QWidget()
        self.StandardPlots.setObjectName("StandardPlots")
        self.ShowInfectedCheckBox = QtWidgets.QCheckBox(self.StandardPlots)
        self.ShowInfectedCheckBox.setGeometry(QtCore.QRect(10, 26, 171, 31))
        self.ShowInfectedCheckBox.setObjectName("ShowInfectedCheckBox")
        self.ShortestPathCheckbox = QtWidgets.QCheckBox(self.StandardPlots)
        self.ShortestPathCheckbox.setGeometry(QtCore.QRect(10, 70, 171, 31))
        self.ShortestPathCheckbox.setObjectName("ShortestPathCheckbox")
        self.SpinBoxSource = QtWidgets.QSpinBox(self.StandardPlots)
        self.SpinBoxSource.setGeometry(QtCore.QRect(10, 110, 111, 22))
        self.SpinBoxSource.setObjectName("SpinBoxSource")
        self.SpinBoxSource.setRange(0, 99999)
        self.SpinBoxTarget = QtWidgets.QSpinBox(self.StandardPlots)
        self.SpinBoxTarget.setGeometry(QtCore.QRect(10, 160, 111, 22))
        self.SpinBoxTarget.setObjectName("SpinBoxTarget")
        self.SpinBoxTarget.setRange(0, 99999)
        self.line = QtWidgets.QFrame(self.StandardPlots)
        self.line.setGeometry(QtCore.QRect(10, 90, 171, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.StandardPlots)
        self.line_2.setGeometry(QtCore.QRect(10, 180, 171, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_21 = QtWidgets.QLabel(self.StandardPlots)
        self.label_21.setGeometry(QtCore.QRect(130, 110, 47, 21))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.StandardPlots)
        self.label_22.setGeometry(QtCore.QRect(130, 160, 47, 21))
        self.label_22.setObjectName("label_22")
        self.stackedWidget.addWidget(self.StandardPlots)
        ## Page for clustering coefficient plot and adjacency matrix
        self.MatrixClustering = QtWidgets.QWidget()
        self.MatrixClustering.setObjectName("MatrixClustering")
        self.stackedWidget.addWidget(self.MatrixClustering)

        # Main window
        MainWindow.setCentralWidget(self.centralwidget)
        ## Menubar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        Menu = self.menubar
        self.menubar.setGeometry(QtCore.QRect(0, 0, 878, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionUpload = QtWidgets.QAction(MainWindow)
        self.actionUpload.setObjectName("actionUpload")
        self.menuFile.addAction(self.actionUpload)
        self.menubar.addAction(self.menuFile.menuAction())
        Menu.triggered.connect(self.MenuAction)     #Menu tree actions
        self.menuFile.addAction(self.actionUpload)  #Add upload action

        self.retranslateUi(MainWindow)
        self.PlotOptionsStackedWidget.setCurrentIndex(1)
        self.TabBrowserWidget.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_13.setText(_translate("MainWindow", "Spring constant"))
        self.label_23.setText(_translate("MainWindow", "Spring length"))
        self.BarnesHutOverlap.setText(_translate("MainWindow", "Avoid overlap"))
        self.label_6.setText(_translate("MainWindow", "Gravity constant"))
        self.label_7.setText(_translate("MainWindow", "Central gravity"))
        self.label_14.setText(_translate("MainWindow", "Damping"))
        self.label_2.setText(_translate("MainWindow", "Node distance"))
        self.label_3.setText(_translate("MainWindow", "Central gravity"))
        self.label_4.setText(_translate("MainWindow", "Spring length"))
        self.label_5.setText(_translate("MainWindow", "Spring constant"))
        self.label_15.setText(_translate("MainWindow", "Damping"))
        self.label_11.setText(_translate("MainWindow", "Node distance"))
        self.label_12.setText(_translate("MainWindow", "Damping"))
        self.label_8.setText(_translate("MainWindow", "Spring length"))
        self.label_9.setText(_translate("MainWindow", "Spring constant"))
        self.label_10.setText(_translate("MainWindow", "Central gravity"))
        self.label_16.setText(_translate("MainWindow", "Spring constant"))
        self.label_17.setText(_translate("MainWindow", "Gravity constant"))
        self.label_18.setText(_translate("MainWindow", "Central gravity"))
        self.label_19.setText(_translate("MainWindow", "Damping"))
        self.label_20.setText(_translate("MainWindow", "Spring length"))
        self.ForcedAtlasOverlap.setText(_translate("MainWindow", "Avoid overlap"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Hierarchical repulsion"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Repulsion"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Barnes hut"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Forced atlas"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Adjacency matrix"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Clustering coefficient"))
        self.comboBox.setItemText(6, _translate("MainWindow", "DH layout"))
        self.comboBox.setItemText(7, _translate("MainWindow", "Sphere layout"))
        self.comboBox.setItemText(8, _translate("MainWindow", "Sugiyama layout"))
        self.comboBox.setItemText(9, _translate("MainWindow", "MDS layout"))
        self.comboBox.setItemText(10, _translate("MainWindow", "LGL layout"))
        self.comboBox.setItemText(11, _translate("MainWindow", "KK layout"))
        self.comboBox.setItemText(12, _translate("MainWindow", "Graphopt layout"))
        self.comboBox.setItemText(13, _translate("MainWindow", "FR layout"))
        self.comboBox.setItemText(14, _translate("MainWindow", "Geo layout"))
        self.comboBox.setItemText(15, _translate("MainWindow", "Star layout"))
        self.comboBox.setItemText(16, _translate("MainWindow", "Grid layout"))
        self.comboBox.setItemText(17, _translate("MainWindow", "Tree layout"))
        self.comboBox.setItemText(18, _translate("MainWindow", "Gem layout"))
        self.comboBox.setItemText(19, _translate("MainWindow", "Circle layout"))
        self.comboBox.setItemText(20, _translate("MainWindow", "Vis"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab), _translate("MainWindow", "Plot"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab_2), _translate("MainWindow", "Raw data"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab_3), _translate("MainWindow", "Nodes"))
        self.label.setText(_translate("MainWindow", "Timestep: 0"))
        self.ApplyChanges.setText(_translate("MainWindow", "Apply changes"))
        self.ShowInfectedCheckBox.setText(_translate("MainWindow", "Show infected"))
        self.ShortestPathCheckbox.setText(_translate("MainWindow", "Shortest path"))
        self.label_21.setText(_translate("MainWindow", "Source"))
        self.label_22.setText(_translate("MainWindow", "Target"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionUpload.setText(_translate("MainWindow", "Upload"))

    def BtnAction(self):
        '''
        Do the actions when the button is pressed
        '''
        # First we update the dataframe, according to the timestep
        timestep_value = self.timestep_slider.value()
        self.plot_df = self.df[self.df['timestep'] <= timestep_value]

        # Then we generate the table
        model = DataFrameModel(self.plot_df)
        self.tableView.setModel(model)

        # Generate the graph
        self.UpdatePlotGraph() # Update or generate the test.html, which contais the graph that will be injected

        # Plot the graph
        path = os.getcwd() # Get the absolute path from os, because it is better for PyQT5
        url = QtCore.QUrl.fromLocalFile(path + '/test.html')
        self.PlotWidget.load(url)

    def getGraphType(self):
        '''
        Sets the stacked widgets page according to which plot was selected
            Graph plots receive their parameters in the stacked widget on the left and statistics on the right;
            Some graphs don't receive anything.
        '''
        #print(self.comboBox.currentText())
        if self.comboBox.currentText() == 'Hierarchical repulsion':
            self.PlotOptionsStackedWidget.setCurrentIndex(1)  # sets the configurations available
            self.stackedWidget.setCurrentIndex(0)

        elif self.comboBox.currentText() == 'Repulsion':
            self.PlotOptionsStackedWidget.setCurrentIndex(2)  # sets the configurations available
            self.stackedWidget.setCurrentIndex(0)

        elif self.comboBox.currentText() == 'Barnes hut':
            self.PlotOptionsStackedWidget.setCurrentIndex(0)  # sets the configurations available
            self.stackedWidget.setCurrentIndex(0)

        elif self.comboBox.currentText() == 'Forced atlas':
            self.PlotOptionsStackedWidget.setCurrentIndex(3)  # sets the configurations available
            self.stackedWidget.setCurrentIndex(0)

        elif self.comboBox.currentText() == 'Clustering coefficient':
            self.stackedWidget.setCurrentIndex(1)
            self.PlotOptionsStackedWidget.setCurrentIndex(4)

        elif self.comboBox.currentText() == 'Ajcacency matrix':
            self.stackedWidget.setCurrentIndex(1)
            self.PlotOptionsStackedWidget.setCurrentIndex(4)

        else:
            self.PlotOptionsStackedWidget.setCurrentIndex(4)  # sets the configurations available
            self.stackedWidget.setCurrentIndex(0)

    def UpdatePlotGraph(self):
        
        timestep_value = self.timestep_slider.value()
        self.plot_df = self.df[self.df['timestep'] <= timestep_value]
        self.data = self.plot_df.to_numpy()
        index = np.argmax(self.data[:, 0] == 100, axis=0)
        index = index if index > 0 else len(self.data)
        self.data = self.data[:index]
        data = self.data
        self.edgeList = self.data[:index, 1:3].tolist()
        self.edgeList = [sorted(x) for x in self.edgeList]

        self.edf = pd.DataFrame(self.edgeList)
        self.weight_edf = self.edf.pivot_table(index=[0, 1], aggfunc='size')

        weight_edges = []
        for x in data:
            tmp = sorted([x[1], x[2]])
            weight_edges.append(self.weight_edf[tmp[0], tmp[1]])
        self.weight_edges = np.asarray(weight_edges)

        G = nx.Graph()
        for i in range(len(self.weight_edf.values)):
            G.add_edge(self.weight_edf.index[i][0], self.weight_edf.index[i][1], weight=self.weight_edf.iloc[i])
        self.graph = G.copy()
        self.matrix = nx.to_numpy_matrix(G)
        self.nodes = list(G)

        rpy2.robjects.numpy2ri.activate()
        utils.chooseCRANmirror(ind=1)
        person1 = ro.IntVector(data[:index, 1])
        person2 = ro.IntVector(data[:index, 2])

        set_vars(data, index, self.plot_df, self.edgeList, self.graph)

        # Here we generate the table with the nodes properties
        self.df_nodes = pd.DataFrame()
        self.df_nodes['Node'] = list(G)
        self.df_nodes['Infected'] = get_infected()
        self.df_nodes['Clustering coef'] = nx.clustering(G).values()
        model = DataFrameModel(self.df_nodes)
        self.tableView_2.setModel(model)

        if self.comboBox.currentText() == 'Hierarchical repulsion':
            node_distance = self.hierarchical_node_distance.value()
            central_gravity = self.hierarchical_gravity.value()
            spring_length = self.hierarchical_springlength.value()
            spring_constant = self.hierarchical_springconstant.value()
            damping = self.hierarchical_damping.value()
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
                print(path)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            hierarchicalRepulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
                          path=path, weight=None, nodeDistance=node_distance, central_gravity=central_gravity, spring_length=spring_length,
                          spring_constant=spring_constant, damping=damping)

        elif self.comboBox.currentText() == 'Repulsion':
            node_distance = self.repulsion_node_distance.value()
            central_gravity = self.repulsion_gravity.value()
            spring_length = self.repulsion_springlength.value()
            spring_constant = self.repulsion_springconstant.value()
            damping = self.repulsion_damping.value()
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            repulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
                          path=path, weight=None, nodeDistance=node_distance, central_gravity=central_gravity, spring_length=spring_length,
                          spring_constant=spring_constant, damping=damping)
            
        elif self.comboBox.currentText() == 'Barnes hut':
            gravity_constant = self.BarnesHut_GravityCtt.value()
            central_gravity = self.BarnesHut_CentralGravity.value()
            spring_length = self.BarnesHut_SpringLength.value()
            spring_constant = self.BarnesHut_SpringCtt.value()
            damping = self.BarnesHut_Damping.value()
            avoidOverlap = int(self.BarnesHutOverlap.isChecked())
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            barnesHut(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None, gravity_constant=gravity_constant, central_gravity=central_gravity, spring_length=spring_length,
              spring_constant=spring_constant, damping=damping, avoidOverlap=avoidOverlap)

        elif self.comboBox.currentText() == 'Forced atlas':
            gravity_constant = self.ForcedAtlas_GravityCtt.value()
            central_gravity = self.ForcedAtlas_CentralGravity.value()
            spring_length = self.ForcedAtlas_SpringLength.value()
            spring_constant = self.ForcedAtlas_SpringCtt.value()
            damping = self.ForcedAtlas_Damping.value()
            avoidOverlap = int(self.ForcedAtlasOverlap.isChecked())
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            forced_atlas(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
                 path=path, weight=None, gravity_constant=gravity_constant, central_gravity=central_gravity, spring_length=spring_length,
                 spring_constant=spring_constant, damping=damping, avoidOverlap=avoidOverlap)

        elif self.comboBox.currentText() == 'Adjacency matrix':
            adjacency_matrix(self.nodes, self.matrix)

        #elif self.comboBox.currentText() == 'Clustering coefficient':

        elif self.comboBox.currentText() == 'DH layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            dh_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'Sphere layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            sphere_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'Sugiyama layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            sugiyama_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'MDS layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            mds_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'LGL layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            lgl_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'KK layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            kk_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'Graphopt layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            graphopt_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'FR layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            fr_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

        elif self.comboBox.currentText() == 'Star layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            star_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Grid layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            grid_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Geo layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            geo_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Tree layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            tree_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Gem layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            gem_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Circle layout':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            circle_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)
        
        elif self.comboBox.currentText() == 'Vis':
            path = None
            infected = None
            if self.ShortestPathCheckbox.isChecked():
                path = [self.SpinBoxSource.value(), self.SpinBoxTarget.value()]
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected()
            visGraph(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=None,
              path=path, weight=None)

    def MenuAction(self, action):
        '''
        Actions for the menu
        '''
        if action.text() == "Upload":
            self.openFileNameDialog()

    def openFileNameDialog(self):
        '''
        Open the File Dialog window and gets the .csv file name
        '''
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(MainWindow,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv)", options=options)
        if fileName:
            self.df = pd.read_csv(fileName)
            minimum_timestep = self.df['timestep'].min()
            maximum_timestep = self.df['timestep'].max()
            self.timestep_slider.setMinimum(minimum_timestep)
            self.timestep_slider.setMaximum(maximum_timestep)
            timestep_value = maximum_timestep
            self.ApplyChanges.setEnabled(True)

    def updateSliderLabel(self, value):
        '''
        Update the label above the slider (label taht shows the timestep value)
        '''
        self.label.setText("Timestep: " + str(value))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
