from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(873, 588)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #StackedWidget for plot configs
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 30, 161, 481))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.radioButton = QtWidgets.QRadioButton(self.page)
        self.radioButton.setGeometry(QtCore.QRect(10, 50, 82, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.page)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 70, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.checkBox = QtWidgets.QCheckBox(self.page)
        self.checkBox.setGeometry(QtCore.QRect(10, 90, 70, 17))
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.page)
        self.checkBox_2.setGeometry(QtCore.QRect(10, 110, 70, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.comboBox_2 = QtWidgets.QComboBox(self.page_2)
        self.comboBox_2.setGeometry(QtCore.QRect(10, 20, 69, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.stackedWidget.addWidget(self.page_2)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.checkBox_3 = QtWidgets.QCheckBox(self.page_5)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 80, 70, 17))
        self.checkBox_3.setObjectName("checkBox_3")
        self.stackedWidget.addWidget(self.page_5)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(0, 0, 161, 31))

        #Horizontal slider for timestep
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(0, 511, 161, 31))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")

        #TabWidget for table and graph
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(160, 0, 531, 541))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.widget = QtWebEngineWidgets.QWebEngineView(self.tab)
        self.widget.setGeometry(QtCore.QRect(0, 0, 531, 521))
        self.widget.setObjectName("widget")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableView = QtWidgets.QTableView(self.tab_2)
        self.tableView.setGeometry(QtCore.QRect(0, 0, 531, 521))
        self.tableView.setObjectName("tableView")
        self.tabWidget.addTab(self.tab_2, "")
        
        self.stackedWidget_2 = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget_2.setGeometry(QtCore.QRect(690, 0, 181, 541))
        self.stackedWidget_2.setObjectName("stackedWidget_2")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.stackedWidget_2.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.stackedWidget_2.addWidget(self.page_4)
        MainWindow.setCentralWidget(self.centralwidget)

        #Menu
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        Menu = self.menubar
        self.menubar.setGeometry(QtCore.QRect(0, 0, 873, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionUpload = QtWidgets.QAction(MainWindow)
        self.actionUpload.setObjectName("actionUpload")
        Menu.triggered.connect(self.MenuAction)     #Menu tree actions
        self.menuFile.addAction(self.actionUpload)  #Add upload action
        self.menubar.addAction(self.menuFile.menuAction())

        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.currentIndexChanged.connect(self.getGraphType) #Change the graph ONLY if the ComboBox index changed

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def getGraphType(self):
        print(self.comboBox.currentText())
        if self.comboBox.currentText() == 'Directed graph':
            self.stackedWidget.setCurrentIndex(0)
            html = open('networkInteractive1.html').read()
            self.widget.setHtml(html)
        elif self.comboBox.currentText() == 'Radial graph':
            self.stackedWidget.setCurrentIndex(1)
        elif self.comboBox.currentText() == 'Adjacency matrix':
            self.stackedWidget.setCurrentIndex(2)

    def MenuAction(self, action):
        print(action.text())
        if action.text() == "Upload":
            self.openFileNameDialog()

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(MainWindow,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv)", options=options)
        if fileName:
            print(fileName)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton.setText(_translate("MainWindow", "RadioButton"))
        self.radioButton_2.setText(_translate("MainWindow", "RadioButton"))
        self.checkBox.setText(_translate("MainWindow", "CheckBox"))
        self.checkBox_2.setText(_translate("MainWindow", "CheckBox"))
        self.checkBox_3.setText(_translate("MainWindow", "CheckBox"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Directed graph"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Radial graph"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Adjacency matrix"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionUpload.setText(_translate("MainWindow", "Upload"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

