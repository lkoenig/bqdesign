#!/usr/bin/env python3
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph.parametertree.parameterTypes as pTypes
import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import logging
import sys
from scipy import signal
import os
from biquad_design import BIQUAD_DESIGN_LIBRARY

logging.basicConfig()

SAMPLERATE = 48000
NPOINTS = 8192


class FilterParameter(pTypes.GroupParameter):
    coefficients_changed = QtCore.Signal()

    def __init__(self, **opts):
        super().__init__(**opts)


class InductanceFilterParameter(FilterParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        super().__init__(**opts)

        self.addChild(
            {
                'name': 'R',
                'type': 'float',
                'value': 1.0,
                'suffix': 'Ohm',
                'siPrefix': False,
            })
        self.addChild(
            {
                'name': 'L',
                'type': 'float',
                'value': 0.0,
                'suffix': 'H',
                'siPrefix': False,
            })
        self.addChild(
            {
                'name': 'Rleak',
                'type': 'float',
                'value': 20,
                'suffix': 'Ohm',
                'siPrefix': False,
            })
        self.R = self.param('R')
        self.L = self.param('L')
        self.Rleak = self.param('Rleak')
        self.R.sigValueChanged.connect(self.design_filter)
        self.L.sigValueChanged.connect(self.design_filter)
        self.Rleak.sigValueChanged.connect(self.design_filter)

    def design_filter(self):
        R = self.R.value()
        Rleak = self.Rleak.value()
        L = self.L.value()
        Zs = signal.lti([(R + Rleak) * L, R * Rleak], [L, Rleak])
        Zd = Zs.to_discrete(1 / SAMPLERATE)
        self.b = Zd.num
        self.a = Zd.den
        self.sos = signal.tf2sos(self.b, self.a)
        self.coefficients_changed.emit()


class DigitalFilterParameter(FilterParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        super().__init__(**opts)

        self.addChild(
            {
                'name': 'numerator',
                'type': 'text',
                'value': '1.0, 0.0, 0.0',
                'siPrefix': False,
            })
        self.addChild(
            {
                'name': 'denominator',
                'type': 'text',
                'value': '1.0, 0.0, 0.0',
                'siPrefix': False
            })
        self.numerator = self.param('numerator')
        self.denominator = self.param('denominator')
        self.numerator.sigValueChanged.connect(self.design_filter)
        self.denominator.sigValueChanged.connect(self.design_filter)
        

    def design_filter(self):
        self.a = np.fromstring(self.denominator.value(), dtype=np.float64)
        self.b = np.fromstring(self.numerator.value(), dtype=np.float64)
        self.sos = signal.tf2sos(self.b, self.a)
        self.coefficients_changed.emit()


class BiquadParameter(FilterParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        super().__init__(**opts)

        self.addChild(
            {
                'name': 'type',
                'type': 'list',
                'value': 'parametric eq',
                'siPrefix': True,
                'limits': BIQUAD_DESIGN_LIBRARY.keys()
            })
        self.addChild(
            {
                'name': 'Q',
                'type': 'float',
                'value': np.sqrt(2) / 2,
                'siPrefix': True
            })
        self.addChild(
            {
                'name': 'fc',
                'type': 'float',
                'value': 100,
                'suffix': 'Hz',
                'siPrefix': True
            })
        self.addChild(
            {
                'name': 'gain',
                'type': 'float',
                'value': 0,
                'suffix': 'dB',
                'siPrefix': True
            })
        self.addChild(
            {
                'name': 'linear gain',
                'type': 'float',
                'value': 1.0,
                'min': 0.0,
                'siPrefix': False,
            })

        self.filter_type = self.param('type')
        self.Q = self.param('Q')
        self.fc = self.param('fc')
        self.gain = self.param('gain')
        self.linear_gain = self.param('linear gain')

        self.filter_type.sigValueChanged.connect(self.design_filter)
        self.Q.sigValueChanged.connect(self.design_filter)
        self.fc.sigValueChanged.connect(self.design_filter)
        self.gain.sigValueChanged.connect(self.on_gain_changed)
        self.linear_gain.sigValueChanged.connect(self.on_linear_gain_changed)
        self.linear_gain.sigValueChanged.connect(self.design_filter)

        self.b = np.array([1.0, 0, 0])
        self.a = np.array([1.0, 0, 0])
        self.design_filter()

    def on_linear_gain_changed(self):
        self.gain.setValue(20 * np.log10(self.linear_gain.value()))

    def on_gain_changed(self):
        self.linear_gain.setValue(10**(self.gain.value()/20.0))

    def design_filter(self):
        """
        This filter is used to design Biquad Filters

        This creates the A and B coefficients for Biquad IIR filters as used in CSPL
        """

        f0 = self.fc.value()
        Q = self.Q.value()
        gain = self.linear_gain.value()

        biquadType = self.filter_type.value()

        design_function = BIQUAD_DESIGN_LIBRARY[biquadType]
        b_coefficients, a_coefficients = design_function(
            gain, f0, Q, SAMPLERATE)

        # if the filter design results in a numerator and denominator that are equal, then just implement the passthrough filter
        if np.array_equal(b_coefficients, a_coefficients):
            bCoeff = np.array([1., 0., 0.])
            aCoeff = np.array([1., 0., 0.])

        self.a = np.array(a_coefficients) / a_coefficients[0]
        self.b = np.array(b_coefficients) / a_coefficients[0]

        assert len(self.a) == 3
        assert len(self.b) == 3

        self.sos = signal.tf2sos(self.b, self.a)
        self.coefficients_changed.emit()


class FilterCascadeParameter(FilterParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"

        self.filter_types = {
            "Biquad": BiquadParameter,
            "Inductance": InductanceFilterParameter,
            # "Custom": DigitalFilterParameter,
        }

        opts['addList'] = list(self.filter_types.keys())
        super().__init__(**opts)
        for child in self.children():
            if isinstance(child, FilterParameter):
                child.coefficients_changed.connect(self.update_fiter)

    def addNew(self, filter_type="Biquad"):
        filter_type_cls = self.filter_types[filter_type]
        child = self.addChild(filter_type_cls(name="%s %d" % (filter_type,
                                                              len(self.childs)+1), removable=True, renamable=True))
        if isinstance(child, FilterParameter):
            child.coefficients_changed.connect(self.update_fiter)
    
    def update_fiter(self):
        self.coefficients_changed.emit()


class BiquadDesigner(QtGui.QWidget):
    def __init__(self):
        super().__init__()
        self._colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self._color_index = 0
        self.filter_parameters = FilterCascadeParameter(name='Filter Cascade', children=[
            BiquadParameter(name="Biquad 1", removable=True, renamable=True)])

        self.parameter_tree = ParameterTree()
        self.parameter_tree.setParameters(self.filter_parameters, showTop=True)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        self.plot_widget = pyqtgraph.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        self.plot.setTitle("Magnitude response")
        self.plot.setLabel("bottom", "Frequency", "Hz")
        self.plot.setXRange(20, SAMPLERATE/2)
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()

        self.normalized_frequencies = np.linspace(1/NPOINTS, 0.5, NPOINTS)
        magnitude = np.ones(self.normalized_frequencies.shape)
        self.total_response_magnitude = self.plot.plot(
            x=self.normalized_frequencies*SAMPLERATE, y=magnitude, name="Total", pen=self.get_next_pen())

        toolbar = QtGui.QToolBar("Main toolbar")
        load_data_button = QtGui.QAction("Load data", self)
        load_data_button.triggered.connect(self.load_data_triggered)
        toolbar.addAction(load_data_button)

        save_button = QtGui.QAction("Save", self)
        save_button.triggered.connect(self.save_sos_coefficients)
        toolbar.addAction(save_button)

        refresh_button = QtGui.QAction("Refresh", self)
        refresh_button.triggered.connect(self.update_frequency_response)
        toolbar.addAction(refresh_button)

        layout.addWidget(toolbar)
        layout.addWidget(self.parameter_tree)
        layout.addWidget(self.plot_widget)
        self.update_frequency_response()
        self.resize(800, 800)
        self.show()
        self.filter_parameters.coefficients_changed.connect(
            self.update_frequency_response)

    def get_next_pen(self):
        c = self._colors[self._color_index]
        self._color_index = (self._color_index + 1) % len(self._colors)
        return pyqtgraph.mkPen(c)


    def update_frequency_response(self):
        self.sos = self.get_second_order_sections()
        w, h = signal.sosfreqz(self.sos, self.normalized_frequencies * np.pi * 2)
        magnitude = np.abs(h)
        self.total_response_magnitude.setData(
            x=self.normalized_frequencies * SAMPLERATE, y=magnitude)

    def get_second_order_sections(self):
        sos = []
        for f in self.filter_parameters:
            if hasattr(f, "sos"):
                sos.append(f.sos)
        if len(sos) == 0:
            sos = [[1, 0, 0, 1, 0, 0]]
        return np.vstack(sos)

    def load_data_triggered(self):
        data_filename, file_type = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                                                                     os.getcwd(), "Data files (*.csv *.txt)")
        data = np.loadtxt(data_filename)
        if data.shape[1] == 2:
            self.plot.plot(x=data[:, 0], y=data[:, 1],
                           name=str(os.path.basename(data_filename)),
                           pen=self.get_next_pen())

    def save_sos_coefficients(self):
        def textproto_sos_save(filename, data):
            if data.shape[1] != 6:
                raise TypeError("All sections should be 6 coefficients")
            with open(filename, "w") as fid:
                for section in data:
                    fid.write(
                        "biquad: {{b0: {0:.6e} b1: {1:.6e} b2: {2:.6e} a1: {4:.6e} a2: {5:.6e} }}\n".format(*section))

        filters = {
            "Data files (*.txt)": lambda filename, data: np.savetxt(filename, data,
                                                                    header="# Second order secions coefficients"),
            "Textproto (*.textproto)": textproto_sos_save,
        }
        sos_filename, file_type = QtGui.QFileDialog.getSaveFileName(self, 'Save file',
                                                                    os.getcwd(), ';;'.join(filters.keys()))
        if sos_filename == "":
            return
        sos = self.get_second_order_sections()
        save_method = filters[file_type]
        save_method(sos_filename, sos)


app = QtGui.QApplication(sys.argv)
ex = BiquadDesigner()
sys.exit(app.exec_())

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
