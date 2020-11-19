from PySide2 import QtWidgets

import os

import numpy as np


class DataCurveWidget(QtWidgets.QWidget):
    '''Widget to add plot curve
    '''

    def __init__(self, filename, plot_area):
        super().__init__()
        self.data = np.loadtxt(filename)
        if self.data.shape[1] == 2:
            self.plot_data = plot_area.plot(
                x=self.data[:, 0], y=self.data[:, 1],
                name=str(os.path.basename(
                    filename)))


class DataCurveListWidget(QtWidgets.QWidget):
    '''Widget to hold a list of data curve
    '''

    def __init__(self, plot_area):
        """
        docstring
        """
        super().__init__()
        self.plot_area = plot_area
        self.layout = QtWidgets.QVBoxLayout()
        toolbar = QtWidgets.QToolBar("Main toolbar")
        self.layout.addWidget(toolbar)

        self.setLayout(self.layout)
        self.resize(100, 100)

        load_data_button = QtWidgets.QAction("Load data", self)
        load_data_button.triggered.connect(self.load_data_triggered)
        toolbar.addAction(load_data_button)

        self.last_folder = os.getcwd()

    def update_last_folder(self, filename):
        self.last_folder = os.path.dirname(filename)

    def load_data_triggered(self):
        data_filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                                 self.last_folder, "Data files (*.csv *.txt)")
        if data_filename == "":
            return
        curve_widget = DataCurveWidget(data_filename, self.plot_area)
        self.layout.addWidget(curve_widget)
        self.update_last_folder(data_filename)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = DataCurveListWidget()
    sys.exit(app.exec_())
