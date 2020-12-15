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

os.environ['R_HOME'] = "C:/Program Files/R/R-3.6.2"
os.environ["PATH"] += os.pathsep + "C:/Program Files/R/R-3.6.2/bin/x64/"
os.environ["PATH"] += os.pathsep + "C:/Program Files/R/R-3.6.2/"

import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

base = rpackages.importr("base")
utils = rpackages.importr("utils")

rpy2.robjects.numpy2ri.activate()
utils.chooseCRANmirror(ind=1)

ro.r(f'''
    #install.packages("igraph")
    #install.packages("visNetwork")
    #install.packages("networkD3")
    #install.packages("ggplot2")
    #install.packages("plotly")
    #install.packages("hrbrthemes")
    #install.packages("viridis")
    #install.packages("heatmaply")
    Sys.setenv(RSTUDIO_PANDOC="pandoc")
    s <- Sys.getenv("R.dll")
    ''')

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

        # PLOT OPTIONS STACKED WIDGET
        self.PlotOptionsStackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.PlotOptionsStackedWidget.setGeometry(QtCore.QRect(0, 40, 191, 391))
        self.PlotOptionsStackedWidget.setObjectName("PlotOptionsStackedWidget")
        ## BARNES HUT PAGE (0)
        self.BarnesHut_page = QtWidgets.QWidget()
        self.BarnesHut_page.setObjectName("BarnesHut_page")
        self.BarnesHut_Damping = QtWidgets.QDoubleSpinBox(self.BarnesHut_page)
        self.BarnesHut_Damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.BarnesHut_Damping.setObjectName("BarnesHut_Damping")
        self.BarnesHut_SpringConstantLabel = QtWidgets.QLabel(self.BarnesHut_page)
        self.BarnesHut_SpringConstantLabel.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.BarnesHut_SpringConstantLabel.setObjectName("BarnesHut_SpringConstantLabel")
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
        self.BarnesHut_SpringLengthLabel = QtWidgets.QLabel(self.BarnesHut_page)
        self.BarnesHut_SpringLengthLabel.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.BarnesHut_SpringLengthLabel.setObjectName("BarnesHut_SpringLengthLabel")
        self.BarnesHutOverlap = QtWidgets.QCheckBox(self.BarnesHut_page)
        self.BarnesHutOverlap.setGeometry(QtCore.QRect(10, 360, 161, 17))
        self.BarnesHutOverlap.setObjectName("BarnesHutOverlap")
        self.BarnesHut_GravityConstantLabel = QtWidgets.QLabel(self.BarnesHut_page)
        self.BarnesHut_GravityConstantLabel.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.BarnesHut_GravityConstantLabel.setObjectName("BarnesHut_GravityConstantLabel")
        self.BarnesHut_CentralGravityLabel = QtWidgets.QLabel(self.BarnesHut_page)
        self.BarnesHut_CentralGravityLabel.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.BarnesHut_CentralGravityLabel.setObjectName("BarnesHut_CentralGravityLabel")
        self.BarnesHut_DampingLabel = QtWidgets.QLabel(self.BarnesHut_page)
        self.BarnesHut_DampingLabel.setGeometry(QtCore.QRect(10, 300, 161, 16))
        self.BarnesHut_DampingLabel.setObjectName("BarnesHut_DampingLabel")
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
        ## HIERARCHICAL PLOT PAGE (1)
        self.hierarchical_page = QtWidgets.QWidget()
        self.hierarchical_page.setObjectName("hierarchical_page")
        self.hierarchical_node_distanceLabel = QtWidgets.QLabel(self.hierarchical_page)
        self.hierarchical_node_distanceLabel.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.hierarchical_node_distanceLabel.setObjectName("hierarchical_node_distanceLabel")
        self.hierarchical_node_distance = QtWidgets.QSpinBox(self.hierarchical_page)
        self.hierarchical_node_distance.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.hierarchical_node_distance.setObjectName("hierarchical_node_distance")
        self.hierarchical_gravity = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_gravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.hierarchical_gravity.setObjectName("hierarchical_gravity")
        self.hierarchical_gravityLabel = QtWidgets.QLabel(self.hierarchical_page)
        self.hierarchical_gravityLabel.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.hierarchical_gravityLabel.setObjectName("hierarchical_gravityLabel")
        self.hierarchical_springlength = QtWidgets.QSpinBox(self.hierarchical_page)
        self.hierarchical_springlength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.hierarchical_springlength.setObjectName("hierarchical_springlength")
        self.hierarchical_springlengthLabel = QtWidgets.QLabel(self.hierarchical_page)
        self.hierarchical_springlengthLabel.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.hierarchical_springlengthLabel.setObjectName("hierarchical_springlengthLabel")
        self.hierarchical_springconstant = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_springconstant.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.hierarchical_springconstant.setObjectName("hierarchical_springconstant")
        self.hierarchical_springconstantLabel = QtWidgets.QLabel(self.hierarchical_page)
        self.hierarchical_springconstantLabel.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.hierarchical_springconstantLabel.setObjectName("hierarchical_springconstantLabel")
        self.hierarchical_damping = QtWidgets.QDoubleSpinBox(self.hierarchical_page)
        self.hierarchical_damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.hierarchical_damping.setObjectName("hierarchical_damping")
        self.hierarchical_dampingLabel = QtWidgets.QLabel(self.hierarchical_page)
        self.hierarchical_dampingLabel.setGeometry(QtCore.QRect(10, 300, 161, 16))
        self.hierarchical_dampingLabel.setObjectName("hierarchical_dampingLabel")
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
        ## REPULSION PAGE (2)
        self.Repulsion_page = QtWidgets.QWidget()
        self.Repulsion_page.setObjectName("Repulsion_page")
        self.repulsion_node_distanceLabel = QtWidgets.QLabel(self.Repulsion_page)
        self.repulsion_node_distanceLabel.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.repulsion_node_distanceLabel.setObjectName("repulsion_node_distanceLabel")
        self.repulsion_node_distance = QtWidgets.QSpinBox(self.Repulsion_page)
        self.repulsion_node_distance.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.repulsion_node_distance.setObjectName("repulsion_node_distance")
        self.repulsion_springconstant = QtWidgets.QDoubleSpinBox(self.Repulsion_page)
        self.repulsion_springconstant.setGeometry(QtCore.QRect(10, 250, 161, 22))
        self.repulsion_springconstant.setObjectName("repulsion_springconstant")
        self.repulsion_springlength = QtWidgets.QSpinBox(self.Repulsion_page)
        self.repulsion_springlength.setGeometry(QtCore.QRect(10, 180, 161, 22))
        self.repulsion_springlength.setObjectName("repulsion_springlength")
        self.repulsion_springlengthLabel = QtWidgets.QLabel(self.Repulsion_page)
        self.repulsion_springlengthLabel.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.repulsion_springlengthLabel.setObjectName("repulsion_springlengthLabel")
        self.repulsion_springconstantLabel = QtWidgets.QLabel(self.Repulsion_page)
        self.repulsion_springconstantLabel.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.repulsion_springconstantLabel.setObjectName("repulsion_springconstantLabel")
        self.repulsion_gravityLabel = QtWidgets.QLabel(self.Repulsion_page)
        self.repulsion_gravityLabel.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.repulsion_gravityLabel.setObjectName("repulsion_gravityLabel")
        self.repulsion_gravity = QtWidgets.QDoubleSpinBox(self.Repulsion_page)
        self.repulsion_gravity.setGeometry(QtCore.QRect(10, 110, 161, 22))
        self.repulsion_gravity.setObjectName("repulsion_gravity")
        self.repulsion_dampingLabel = QtWidgets.QLabel(self.Repulsion_page)
        self.repulsion_dampingLabel.setGeometry(QtCore.QRect(10, 300, 161, 16))
        self.repulsion_dampingLabel.setObjectName("repulsion_dampingLabel")
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
        ## FORCED ATLAS PAGE (3)
        self.ForcedAtlas_page = QtWidgets.QWidget()
        self.ForcedAtlas_page.setObjectName("ForcedAtlas_page")
        self.ForcedAtlas_Damping = QtWidgets.QDoubleSpinBox(self.ForcedAtlas_page)
        self.ForcedAtlas_Damping.setGeometry(QtCore.QRect(10, 320, 161, 22))
        self.ForcedAtlas_Damping.setObjectName("ForcedAtlas_Damping")
        self.ForcedAtlas_SpringCttLabel = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.ForcedAtlas_SpringCttLabel.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.ForcedAtlas_SpringCttLabel.setObjectName("ForcedAtlas_SpringCttLabel")
        self.ForcedAtlas_GravityCttLabel = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.ForcedAtlas_GravityCttLabel.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.ForcedAtlas_GravityCttLabel.setObjectName("ForcedAtlas_GravityCttLabel")
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
        self.ForcedAtlas_Spring = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.ForcedAtlas_Spring.setGeometry(QtCore.QRect(10, 90, 161, 16))
        self.ForcedAtlas_Spring.setObjectName("ForcedAtlas_Spring")
        self.ForcedAtlas_DampingLabel = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.ForcedAtlas_DampingLabel.setGeometry(QtCore.QRect(10, 300, 47, 13))
        self.ForcedAtlas_DampingLabel.setObjectName("ForcedAtlas_DampingLabel")
        self.ForcedAtlas_SpringCttLabel_2 = QtWidgets.QLabel(self.ForcedAtlas_page)
        self.ForcedAtlas_SpringCttLabel_2.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.ForcedAtlas_SpringCttLabel_2.setObjectName("ForcedAtlas_SpringCttLabel_2")
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
        ## NO CONFIG PAGE (4)
        self.NoConfig_page = QtWidgets.QWidget()
        self.NoConfig_page.setObjectName("NoConfig_page")
        self.PlotOptionsStackedWidget.addWidget(self.NoConfig_page)

        # PLOT COMBOBOX
        self.PlotSelectionCombobox = QtWidgets.QComboBox(self.centralwidget)
        self.PlotSelectionCombobox.setGeometry(QtCore.QRect(10, 0, 161, 31))
        self.PlotSelectionCombobox.setObjectName("PlotSelectionCombobox")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.addItem("")
        self.PlotSelectionCombobox.currentIndexChanged.connect(self.getGraphType) #Change the graph tab ONLY if the ComboBox index changed

        # TIMESTEP SLIDER
        self.timestep_slider = QtWidgets.QSlider(self.centralwidget)
        self.timestep_slider.setGeometry(QtCore.QRect(10, 520, 161, 31))
        self.timestep_slider.setOrientation(QtCore.Qt.Horizontal)
        self.timestep_slider.setObjectName("timestep_slider")
        self.TimestepLabel = QtWidgets.QLabel(self.centralwidget)
        self.TimestepLabel.setGeometry(QtCore.QRect(6, 494, 161, 20))
        self.TimestepLabel.setObjectName("TimestepLabel")
        self.timestep_slider.valueChanged.connect(self.updateSliderLabel) # Update the slider label

        # TAB BROWSER
        self.TabBrowserWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.TabBrowserWidget.setGeometry(QtCore.QRect(190, 0, 691, 551))
        self.TabBrowserWidget.setObjectName("TabBrowserWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        ## PLOT WIDGET
        self.PlotWidget = QtWebEngineWidgets.QWebEngineView(self.tab)
        self.PlotWidget.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.PlotWidget.setObjectName("PlotWidget")
        self.TabBrowserWidget.addTab(self.tab, "")
        ## RAW DATA TABLE WIDGET
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableView = QtWidgets.QTableView(self.tab_2)
        self.tableView.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.tableView.setObjectName("tableView")
        self.TabBrowserWidget.addTab(self.tab_2, "")
        ## NODE TABLE WIDGET
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tableView_2 = QtWidgets.QTableView(self.tab_3)
        self.tableView_2.setGeometry(QtCore.QRect(0, 0, 691, 531))
        self.tableView_2.setObjectName("tableView_2")
        self.TabBrowserWidget.addTab(self.tab_3, "")

        # APPLY CHANGES BUTTON
        self.ApplyChanges = QtWidgets.QPushButton(self.centralwidget)
        self.ApplyChanges.setGeometry(QtCore.QRect(14, 452, 161, 31))
        self.ApplyChanges.setObjectName("ApplyChanges")
        self.ApplyChanges.setEnabled(False) # The button is just enabled when the user select a file
        self.ApplyChanges.clicked.connect(self.BtnAction) # If the button is pressed, then go to the action

        # STACKED WIDGET FOR PLOT FILTERS
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(880, 40, 191, 181))
        self.stackedWidget.setObjectName("stackedWidget")
        ## SHORTEST PATH PAGE (0)
        self.ShortestPath = QtWidgets.QWidget()
        self.ShortestPath.setObjectName("ShortestPath")
        self.ShortestPathHead = QtWidgets.QLabel(self.ShortestPath)
        self.ShortestPathHead.setGeometry(QtCore.QRect(10, 0, 111, 31))
        self.ShortestPathHead.setObjectName("ShortestPathHead")
        self.ShortestPathSource = QtWidgets.QSpinBox(self.ShortestPath)
        self.ShortestPathSource.setGeometry(QtCore.QRect(10, 70, 111, 22))
        self.ShortestPathSource.setObjectName("ShortestPathSource")
        self.ShortestPathTarget = QtWidgets.QSpinBox(self.ShortestPath)
        self.ShortestPathTarget.setGeometry(QtCore.QRect(10, 140, 111, 22))
        self.ShortestPathTarget.setObjectName("ShortestPathTarget")
        self.ShortestPathSourceLabel = QtWidgets.QLabel(self.ShortestPath)
        self.ShortestPathSourceLabel.setGeometry(QtCore.QRect(10, 50, 111, 21))
        self.ShortestPathSourceLabel.setObjectName("ShortestPathSourceLabel")
        self.ShortestPathTargetLabel = QtWidgets.QLabel(self.ShortestPath)
        self.ShortestPathTargetLabel.setGeometry(QtCore.QRect(10, 120, 111, 21))
        self.ShortestPathTargetLabel.setObjectName("ShortestPathTargetLabel")
        self.ShortestPathTarget.setRange(0, 9999)
        self.ShortestPathSource.setRange(0, 9999)
        self.stackedWidget.addWidget(self.ShortestPath)
        ## K-CORE PAGE (1)
        self.Kcore = QtWidgets.QWidget()
        self.Kcore.setObjectName("Kcore")
        self.Kcoreshellplot = QtWidgets.QLabel(self.Kcore)
        self.Kcoreshellplot.setGeometry(QtCore.QRect(10, 50, 151, 16))
        self.Kcoreshellplot.setObjectName("Kcoreshellplot")
        self.Kshell = QtWidgets.QSpinBox(self.Kcore)
        self.Kshell.setGeometry(QtCore.QRect(10, 70, 111, 21))
        self.Kshell.setObjectName("Kshell")
        self.KcoreHead = QtWidgets.QLabel(self.Kcore)
        self.KcoreHead.setGeometry(QtCore.QRect(10, 0, 111, 31))
        self.KcoreHead.setObjectName("KcoreHead")
        self.Kshell.setRange(0, 9999)
        self.stackedWidget.addWidget(self.Kcore)
        ## DEPTH FILTER PAGE (2)
        self.DepthFilter = QtWidgets.QWidget()
        self.DepthFilter.setObjectName("DepthFilter")
        self.DepthDepth = QtWidgets.QSpinBox(self.DepthFilter)
        self.DepthDepth.setGeometry(QtCore.QRect(10, 140, 111, 22))
        self.DepthDepth.setObjectName("DepthDepth")
        self.DepthSource = QtWidgets.QSpinBox(self.DepthFilter)
        self.DepthSource.setGeometry(QtCore.QRect(10, 70, 111, 22))
        self.DepthSource.setObjectName("DepthSource")
        self.DepthSourceLabel = QtWidgets.QLabel(self.DepthFilter)
        self.DepthSourceLabel.setGeometry(QtCore.QRect(10, 50, 111, 21))
        self.DepthSourceLabel.setObjectName("DepthSourceLabel")
        self.DepthDepthLabel = QtWidgets.QLabel(self.DepthFilter)
        self.DepthDepthLabel.setGeometry(QtCore.QRect(10, 120, 111, 21))
        self.DepthDepthLabel.setObjectName("DepthDepthLabel")
        self.DepthHead = QtWidgets.QLabel(self.DepthFilter)
        self.DepthHead.setGeometry(QtCore.QRect(10, 0, 111, 31))
        self.DepthHead.setObjectName("DepthHead")
        self.stackedWidget.addWidget(self.DepthFilter)
        self.DepthDepth.setRange(0, 9999)
        self.DepthSource.setRange(0, 9999)
        self.stackedWidget.addWidget(self.DepthFilter)
        ## BLANK PAGE (3)
        self.BlankPage = QtWidgets.QWidget()
        self.BlankPage.setObjectName("BlankPage")
        self.stackedWidget.addWidget(self.BlankPage)
        self.PlotFilterCombobox = QtWidgets.QComboBox(self.centralwidget)
        self.PlotFilterCombobox.setGeometry(QtCore.QRect(890, 0, 171, 31))

        # PLOT FILTER COMBOBOX
        self.PlotFilterCombobox.setObjectName("PlotFilterCombobox")
        self.PlotFilterCombobox.addItem("")
        self.PlotFilterCombobox.setItemText(0, "")
        self.PlotFilterCombobox.addItem("")
        self.PlotFilterCombobox.addItem("")
        self.PlotFilterCombobox.addItem("")
        self.PlotFilterCombobox.addItem("")
        self.PlotFilterCombobox.currentIndexChanged.connect(self.getPlotFilter) #Change the graph tab ONLY if the ComboBox index changed

        # SHOW INFECTED NODES CHECKBOX
        self.ShowInfectedCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.ShowInfectedCheckBox.setGeometry(QtCore.QRect(890, 220, 171, 31))
        self.ShowInfectedCheckBox.setObjectName("ShowInfectedCheckBox")

        # MENUBAR
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        Menu = self.menubar
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 21))
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
        self.TabBrowserWidget.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(3)
        self.PlotOptionsStackedWidget.setCurrentIndex(4)
        self.PlotFilterCombobox.setEnabled(False) # disable comboBox for plot filters
        self.PlotSelectionCombobox.setEnabled(False) # disable comboBox for plot filters
        self.ShowInfectedCheckBox.setEnabled(False) # Disabe show infected combobox
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BarnesHut_SpringConstantLabel.setText(_translate("MainWindow", "Spring constant"))
        self.BarnesHut_SpringLengthLabel.setText(_translate("MainWindow", "Spring length"))
        self.BarnesHutOverlap.setText(_translate("MainWindow", "Avoid overlap"))
        self.BarnesHut_GravityConstantLabel.setText(_translate("MainWindow", "Gravity constant"))
        self.BarnesHut_CentralGravityLabel.setText(_translate("MainWindow", "Central gravity"))
        self.BarnesHut_DampingLabel.setText(_translate("MainWindow", "Damping"))
        self.hierarchical_node_distanceLabel.setText(_translate("MainWindow", "Node distance"))
        self.hierarchical_gravityLabel.setText(_translate("MainWindow", "Central gravity"))
        self.hierarchical_springlengthLabel.setText(_translate("MainWindow", "Spring length"))
        self.hierarchical_springconstantLabel.setText(_translate("MainWindow", "Spring constant"))
        self.hierarchical_dampingLabel.setText(_translate("MainWindow", "Damping"))
        self.repulsion_node_distanceLabel.setText(_translate("MainWindow", "Node distance"))
        self.repulsion_springlengthLabel.setText(_translate("MainWindow", "Spring length"))
        self.repulsion_springconstantLabel.setText(_translate("MainWindow", "Spring constant"))
        self.repulsion_gravityLabel.setText(_translate("MainWindow", "Central gravity"))
        self.repulsion_dampingLabel.setText(_translate("MainWindow", "Spring constant"))
        self.ForcedAtlas_SpringCttLabel.setText(_translate("MainWindow", "Spring constant"))
        self.ForcedAtlas_GravityCttLabel.setText(_translate("MainWindow", "Gravity constant"))
        self.ForcedAtlas_Spring.setText(_translate("MainWindow", "Central gravity"))
        self.ForcedAtlas_DampingLabel.setText(_translate("MainWindow", "Damping"))
        self.ForcedAtlas_SpringCttLabel_2.setText(_translate("MainWindow", "Spring length"))
        self.ForcedAtlasOverlap.setText(_translate("MainWindow", "Avoid overlap"))
        self.PlotSelectionCombobox.setItemText(0, _translate("MainWindow", "Hierarchical repulsion"))
        self.PlotSelectionCombobox.setItemText(1, _translate("MainWindow", "Repulsion"))
        self.PlotSelectionCombobox.setItemText(2, _translate("MainWindow", "Barnes hut"))
        self.PlotSelectionCombobox.setItemText(3, _translate("MainWindow", "Forced atlas"))
        self.PlotSelectionCombobox.setItemText(4, _translate("MainWindow", "Adjacency matrix"))
        self.PlotSelectionCombobox.setItemText(5, _translate("MainWindow", "Clustering coefficient"))
        self.PlotSelectionCombobox.setItemText(6, _translate("MainWindow", "DH layout"))
        self.PlotSelectionCombobox.setItemText(7, _translate("MainWindow", "Sphere layout"))
        self.PlotSelectionCombobox.setItemText(8, _translate("MainWindow", "Sugiyama layout"))
        self.PlotSelectionCombobox.setItemText(9, _translate("MainWindow", "MDS layout"))
        self.PlotSelectionCombobox.setItemText(10, _translate("MainWindow", "LGL layout"))
        self.PlotSelectionCombobox.setItemText(11, _translate("MainWindow", "KK layout"))
        self.PlotSelectionCombobox.setItemText(12, _translate("MainWindow", "Graphopt layout"))
        self.PlotSelectionCombobox.setItemText(13, _translate("MainWindow", "FR layout"))
        self.PlotSelectionCombobox.setItemText(14, _translate("MainWindow", "Geo layout"))
        self.PlotSelectionCombobox.setItemText(15, _translate("MainWindow", "Star layout"))
        self.PlotSelectionCombobox.setItemText(16, _translate("MainWindow", "Grid layout"))
        self.PlotSelectionCombobox.setItemText(17, _translate("MainWindow", "Tree layout"))
        self.PlotSelectionCombobox.setItemText(18, _translate("MainWindow", "Gem layout"))
        self.PlotSelectionCombobox.setItemText(19, _translate("MainWindow", "Circle layout"))
        self.PlotSelectionCombobox.setItemText(20, _translate("MainWindow", "Vis"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab), _translate("MainWindow", "Plot"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab_2), _translate("MainWindow", "Raw data"))
        self.TabBrowserWidget.setTabText(self.TabBrowserWidget.indexOf(self.tab_3), _translate("MainWindow", "Node data"))
        self.TimestepLabel.setText(_translate("MainWindow", "Timestep: 0"))
        self.ApplyChanges.setText(_translate("MainWindow", "Apply changes"))
        self.ShortestPathHead.setText(_translate("MainWindow", "Shortest path config"))
        self.ShortestPathSourceLabel.setText(_translate("MainWindow", "Source node"))
        self.ShortestPathTargetLabel.setText(_translate("MainWindow", "Target node"))
        self.Kcoreshellplot.setText(_translate("MainWindow", "K-shell to plot"))
        self.KcoreHead.setText(_translate("MainWindow", "K-core config"))
        self.DepthSourceLabel.setText(_translate("MainWindow", "Source node"))
        self.DepthDepthLabel.setText(_translate("MainWindow", "Depth"))
        self.DepthHead.setText(_translate("MainWindow", "Depth filter config"))
        self.PlotFilterCombobox.setItemText(1, _translate("MainWindow", "Shortest path"))
        self.PlotFilterCombobox.setItemText(2, _translate("MainWindow", "K-core"))
        self.PlotFilterCombobox.setItemText(3, _translate("MainWindow", "Depth"))
        self.PlotFilterCombobox.setItemText(4, _translate("MainWindow", "Cluster encounter coordinates"))
        self.ShowInfectedCheckBox.setText(_translate("MainWindow", "Show infected nodes"))
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

    def getPlotFilter(self):
        '''
        Sets the stacked widget page according to which filter the user wants to apply
        '''
        if self.PlotFilterCombobox.currentText() == '':
            self.stackedWidget.setCurrentIndex(3)
            self.ShowInfectedCheckBox.setEnabled(True)
        elif self.PlotFilterCombobox.currentText() == 'Shortest path':
            self.stackedWidget.setCurrentIndex(0)
            self.ShowInfectedCheckBox.setEnabled(True)
        elif self.PlotFilterCombobox.currentText() == 'K-core':
            self.stackedWidget.setCurrentIndex(1)
            self.ShowInfectedCheckBox.setEnabled(True)
        elif self.PlotFilterCombobox.currentText() == 'Show infected nodes':
            self.stackedWidget.setCurrentIndex(3)
            self.ShowInfectedCheckBox.setEnabled(True)
        elif self.PlotFilterCombobox.currentText() == 'Depth':
            self.stackedWidget.setCurrentIndex(2)
            self.ShowInfectedCheckBox.setEnabled(True)
        elif self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
            self.stackedWidget.setCurrentIndex(3)
            self.ShowInfectedCheckBox.setEnabled(False)
            self.ShowInfectedCheckBox.setChecked(False)

    def getGraphType(self):
        '''
        Sets the stacked widgets page according to which plot was selected
            Graph plots receive their parameters in the stacked widget on the left and statistics on the right;
            Some graphs don't receive anything.
        '''
        #print(self.comboBox.currentText())
        if self.PlotSelectionCombobox.currentText() == 'Hierarchical repulsion':
            self.PlotOptionsStackedWidget.setCurrentIndex(1)
            self.PlotFilterCombobox.setEnabled(True)  # sets the filters available
            if self.PlotFilterCombobox.currentText() != 'Cluster encounter coordinates':
                self.ShowInfectedCheckBox.setEnabled(True) # enable show infected combobox

        elif self.PlotSelectionCombobox.currentText() == 'Repulsion':
            self.PlotOptionsStackedWidget.setCurrentIndex(2)
            self.PlotFilterCombobox.setEnabled(True)  # sets the filters available
            if self.PlotFilterCombobox.currentText() != 'Cluster encounter coordinates':
                self.ShowInfectedCheckBox.setEnabled(True) # enable show infected combobox

        elif self.PlotSelectionCombobox.currentText() == 'Barnes hut':
            self.PlotOptionsStackedWidget.setCurrentIndex(0)
            self.PlotFilterCombobox.setEnabled(True)  # sets the filters available
            if self.PlotFilterCombobox.currentText() != 'Cluster encounter coordinates':
                self.ShowInfectedCheckBox.setEnabled(True) # enable show infected combobox

        elif self.PlotSelectionCombobox.currentText() == 'Forced atlas':
            self.PlotOptionsStackedWidget.setCurrentIndex(3)
            self.PlotFilterCombobox.setEnabled(True)  # sets the filters available
            if self.PlotFilterCombobox.currentText() != 'Cluster encounter coordinates':
                self.ShowInfectedCheckBox.setEnabled(True) # enable show infected combobox

        elif self.PlotSelectionCombobox.currentText() == 'Clustering coefficient':
            self.PlotOptionsStackedWidget.setCurrentIndex(4)
            self.PlotFilterCombobox.setEnabled(False) # disable comboBox for plot filters
            self.ShowInfectedCheckBox.setEnabled(False) # Disabe show infected combobox
            self.PlotFilterCombobox.setCurrentIndex(0)

        elif self.PlotSelectionCombobox.currentText() == 'Adjacency matrix':
            self.PlotOptionsStackedWidget.setCurrentIndex(4)
            self.PlotFilterCombobox.setEnabled(False) # disable comboBox for plot filters
            self.ShowInfectedCheckBox.setEnabled(False) # disable show infected combobox
            self.PlotFilterCombobox.setCurrentIndex(0)
        else:
            self.PlotOptionsStackedWidget.setCurrentIndex(4)  
            self.PlotFilterCombobox.setEnabled(True)   # sets the filters available
            if self.PlotFilterCombobox.currentText() != 'Cluster encounter coordinates':
                self.ShowInfectedCheckBox.setEnabled(True) # enable show infected combobox


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

        person1 = ro.IntVector(data[:index, 1])
        person2 = ro.IntVector(data[:index, 2])

        set_vars(data, index, self.plot_df, self.edgeList, self.graph)

        # Here we generate the table with the nodes properties
        if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
            person1, person2, group = clusters_coordinates_encounter(timestep_value)
            self.df_nodes = pd.DataFrame()
            self.df_nodes['Node'] = sorted(list(np.unique(np.append(np.unique(person1), np.unique(person2)))))
            self.df_nodes['Infected'] = get_infected(person1, person2)
            self.df_nodes['Cluster'] = group

        else:
            self.df_nodes = pd.DataFrame()
            self.df_nodes['Node'] = sorted(list(np.unique(np.append(np.unique(person1), np.unique(person2)))))
            self.df_nodes['Infected'] = get_infected(person1, person2)
            self.df_nodes['Clustering coef'] = nx.clustering(G).values()

        self.df_nodes.sort_values(by='Node', inplace=True)
        self.df_nodes.reset_index(drop=True, inplace=True)
        model = DataFrameModel(self.df_nodes)
        self.tableView_2.setModel(model)

        if self.PlotSelectionCombobox.currentText() == 'Hierarchical repulsion':
            node_distance = self.hierarchical_node_distance.value()
            central_gravity = self.hierarchical_gravity.value()
            spring_length = self.hierarchical_springlength.value()
            spring_constant = self.hierarchical_springconstant.value()
            damping = self.hierarchical_damping.value()
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            hierarchicalRepulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
                          path=path, weight=None, nodeDistance=node_distance, central_gravity=central_gravity, spring_length=spring_length,
                          spring_constant=spring_constant, damping=damping)

        elif self.PlotSelectionCombobox.currentText() == 'Repulsion':
            node_distance = self.repulsion_node_distance.value()
            central_gravity = self.repulsion_gravity.value()
            spring_length = self.repulsion_springlength.value()
            spring_constant = self.repulsion_springconstant.value()
            damping = self.repulsion_damping.value()
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            repulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
                          path=path, weight=None, nodeDistance=node_distance, central_gravity=central_gravity, spring_length=spring_length,
                          spring_constant=spring_constant, damping=damping)
            
        elif self.PlotSelectionCombobox.currentText() == 'Barnes hut':
            gravity_constant = self.BarnesHut_GravityCtt.value()
            central_gravity = self.BarnesHut_CentralGravity.value()
            spring_length = self.BarnesHut_SpringLength.value()
            spring_constant = self.BarnesHut_SpringCtt.value()
            damping = self.BarnesHut_Damping.value()
            avoidOverlap = int(self.BarnesHutOverlap.isChecked())
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            barnesHut(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None, gravity_constant=gravity_constant, central_gravity=central_gravity, spring_length=spring_length,
              spring_constant=spring_constant, damping=damping, avoidOverlap=avoidOverlap)

        elif self.PlotSelectionCombobox.currentText() == 'Forced atlas':
            gravity_constant = self.ForcedAtlas_GravityCtt.value()
            central_gravity = self.ForcedAtlas_CentralGravity.value()
            spring_length = self.ForcedAtlas_SpringLength.value()
            spring_constant = self.ForcedAtlas_SpringCtt.value()
            damping = self.ForcedAtlas_Damping.value()
            avoidOverlap = int(self.ForcedAtlasOverlap.isChecked())
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            forced_atlas(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
                 path=path, weight=None, gravity_constant=gravity_constant, central_gravity=central_gravity, spring_length=spring_length,
                 spring_constant=spring_constant, damping=damping, avoidOverlap=avoidOverlap)

        elif self.PlotSelectionCombobox.currentText() == 'Adjacency matrix':
            adjacency_matrix(self.nodes, self.matrix)

        elif self.PlotSelectionCombobox.currentText() == 'Clustering coefficient':
            hist_hover(self.df_nodes, 'Clustering coef', bins=30)

        elif self.PlotSelectionCombobox.currentText() == 'DH layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            dh_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'Sphere layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            sphere_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'Sugiyama layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            sugiyama_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'MDS layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            mds_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'LGL layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            lgl_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'KK layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            kk_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'Graphopt layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            graphopt_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'FR layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            fr_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)

        elif self.PlotSelectionCombobox.currentText() == 'Star layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            star_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Grid layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            grid_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Geo layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            geo_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Tree layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            tree_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Gem layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            gem_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Circle layout':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            circle_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
              path=path, weight=None)
        
        elif self.PlotSelectionCombobox.currentText() == 'Vis':
            path = None
            infected = None
            group = None
            if self.PlotFilterCombobox.currentText() == 'Shortest path':
                path = [self.ShortestPathSource.value(), self.ShortestPathTarget.value()]
            if self.PlotFilterCombobox.currentText() == 'K-core':
                k = self.Kshell.value()
                person1, person2 = k_core(self.graph, k=k)
            if self.ShowInfectedCheckBox.isChecked():
                infected = get_infected(person1, person2)
            if self.PlotFilterCombobox.currentText() == 'Depth':
                source = self.DepthSource.value()
                plot_depth = self.DepthDepth.value()
                person1, person2 = depth(self.graph, source=source, depth=plot_depth)
            if self.PlotFilterCombobox.currentText() == 'Cluster encounter coordinates':
                person1, person2, group = clusters_coordinates_encounter(timestep_value)
            visGraph(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=infected, group=group,
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
            self.PlotOptionsStackedWidget.setCurrentIndex(1)
            self.PlotFilterCombobox.setEnabled(True) # disable comboBox for plot filters
            self.PlotSelectionCombobox.setEnabled(True) # disable comboBox for plot filters
            self.ShowInfectedCheckBox.setEnabled(True) # Disabe show infected combobox
            # Then we generate the table
            model = DataFrameModel(self.df)
            self.tableView.setModel(model)

    def updateSliderLabel(self, value):
        '''
        Update the label above the slider (label taht shows the timestep value)
        '''
        self.TimestepLabel.setText("Timestep: " + str(value))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
