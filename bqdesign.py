from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph.parametertree.parameterTypes as pTypes
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import logging

logging.basicConfig()

SAMPLERATE = 48000


def db2mag(db):
    return 10**(-db/20.0)


class BiquadParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.addChild(
            {
                'name': 'type',
                'type': 'list',
                'value': 'parametric eq',
                'siPrefix': True,
                'limits': [
                    'variable Q lp2', 'variable Q hp2'
                    'notch', 'allpass',
                    'parametric eq',
                    'low shelve', 'high shelve',
                    'butterworth lp2', 'butterworth hp2',
                    'first order lp', 'first order hp',
                    'bandpass_0dbpeak',
                ]
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
                'value': 7,
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

        self.filter_type = self.param('type')
        self.Q = self.param('Q')
        self.fc = self.param('fc')
        self.gain = self.param('gain')

        self.filter_type.sigValueChanged.connect(self.design_filter)
        self.Q.sigValueChanged.connect(self.design_filter)
        self.fc.sigValueChanged.connect(self.design_filter)
        self.gain.sigValueChanged.connect(self.design_filter)

        self.b = np.array([1.0, 0, 0])
        self.a = np.array([1.0, 0, 0])
        self.design_filter()

    def design_filter(self):
        """
        This filter is used to design Biquad Filters

        This creates the A and B coefficients for Biquad IIR filters as used in CSPL
        """
        sin = np.sin
        cos = np.cos
        pi = np.pi
        sqrt = np.sqrt
        array = np.array
        sinh = np.sinh
        log = np.log
        fs = SAMPLERATE

        f0 = self.fc.value()
        Q = self.Q.value()
        gainDb = self.gain.value()

        biquadType = self.filter_type.value()
        if biquadType == 'variable Q lp2':
            A = db2mag(gainDb)
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([A * (1 - cos(w0)) / 2, A *
                            (1 - cos(w0)), A * (1 - cos(w0)) / 2])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'variable q hp2':
            A = db2mag(gainDb)
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([A * (1 + cos(w0)) / 2, -A *
                            (1 + cos(w0)), A * (1 + cos(w0)) / 2])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'bandpass':
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([sin(w0) / 2, 0, - sin(w0) / 2])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'notch':
            w0 = 2 * pi * f0 / fs
            if abs(w0 - pi) < 0.000000001:
                # notch filter converges to this:
                bCoeff = array([1, 2, 1])
                aCoeff = array([1, 2, 1])
            else:
                alpha1 = sin(w0) / (2 * Q)
                bCoeff = array([1, - 2 * cos(w0), 1])
                aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'allpass':
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([1 - alpha1, - 2 * cos(w0), 1 + alpha1])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'parametric eq':
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            A = db2mag(gainDb / 2)
            bCoeff = array([1 + alpha1 * A, - 2 * cos(w0), 1 - alpha1 * A])
            aCoeff = array([1 + alpha1 / A, - 2 * cos(w0), 1 - alpha1 / A])
        elif biquadType == 'low shelf':
            w0 = 2 * pi * f0 / fs
            A = db2mag(gainDb / 2)
            alpha1 = sqrt(max(2 + (1 / Q - 1) * (A + 1 / A), 0)) * \
                sin(w0) / 2
            bCoeff = array([A * ((A + 1) - (A - 1) * cos(w0) + 2 * sqrt(A) * alpha1),
                            2 * A * ((A - 1) - (A + 1) * cos(w0)),
                            A * ((A + 1) - (A - 1) * cos(w0) - 2 * sqrt(A) * alpha1)])
            aCoeff = array([(A + 1) + (A - 1) * cos(w0) + 2 * sqrt(A) * alpha1,
                            - 2 * ((A - 1) + (A + 1) * cos(w0)),
                            (A + 1) + (A - 1) * cos(w0) - 2 * sqrt(A) * alpha1])
        elif biquadType == 'high shelf':
            w0 = 2 * pi * f0 / fs
            A = db2mag(gainDb / 2)
            alpha1 = sqrt(max(2 + (1 / Q - 1) * (A + 1 / A), 0)) * \
                sin(w0) / 2
            bCoeff = array([A * ((A + 1) + (A - 1) * cos(w0) + 2 * sqrt(A) * alpha1),
                            - 2 * A * ((A - 1) + (A + 1) * cos(w0)),
                            A * ((A + 1) + (A - 1) * cos(w0) - 2 * sqrt(A) * alpha1)])
            aCoeff = array([(A + 1) - (A - 1) * cos(w0) + 2 * sqrt(A) * alpha1,
                            2 * ((A - 1) - (A + 1) * cos(w0)), (A + 1) - (A - 1) * cos(w0) - 2 * sqrt(A) * alpha1])
        elif biquadType == 'butterworth lp2':
            A = db2mag(gainDb)
            Q = 1 / sqrt(2)
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([A * (1 - cos(w0)) / 2, A *
                            (1 - cos(w0)), A * (1 - cos(w0)) / 2])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'butterworth hp2':
            A = db2mag(gainDb)
            Q = 1 / sqrt(2)
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([A * (1 + cos(w0)) / 2, -A *
                            (1 + cos(w0)), A * (1 + cos(w0)) / 2])
            aCoeff = array([1 + alpha1, - 2 * cos(w0), 1 - alpha1])
        elif biquadType == 'first order lp':
            A = db2mag(gainDb)
            w0 = 2 * pi * f0 / fs
            bCoeff = array([A * (sin(w0)), A * (sin(w0)), 0])
            aCoeff = array([sin(w0) + cos(w0) + 1, (sin(w0) - cos(w0) - 1), 0])
        elif biquadType == 'first order hp':
            A = db2mag(gainDb)
            w0 = 2 * pi * f0 / fs
            bCoeff = array([A * (1 + cos(w0)), - A * (1 + cos(w0)), 0])
            aCoeff = array([sin(w0) + cos(w0) + 1, (sin(w0) - cos(w0) - 1), 0])
        elif biquadType == 'bypass':
            bCoeff = array([1, 0, 0])
            aCoeff = array([1, 0, 0])
        elif biquadType == 'bandpass_0dbpeak':
            w0 = 2 * pi * f0 / fs
            alpha1 = sin(w0) / (2 * Q)
            bCoeff = array([alpha1, 0, -alpha1])
            aCoeff = array([1+alpha1, -2*cos(w0), 1-alpha1])
            gain1 = db2mag(gainDb)
            bCoeff = gain1 * bCoeff
        else:
            raise ValueError(
                'Unrecognized biquad type! Got {}'.format(biquadType))

        # if the filter design results in a numerator and denominator that are equal, then just implement the passthrough filter
        if np.array_equal(bCoeff, aCoeff):
            bCoeff = array([1., 0., 0.])
            aCoeff = array([1., 0., 0.])

        self.a = np.array(aCoeff) / aCoeff[0]
        self.b = np.array(bCoeff) / aCoeff[0]

        assert len(self.a) == 3
        assert len(self.b) == 3


class BiquadCascadeParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self):
        self.addChild(BiquadParameter(name="Biquad %d" % (
            len(self.childs)+1), removable=True, renamable=True))


app = QtGui.QApplication([])

p = BiquadCascadeParameter(name='Biquad Cascade', children=[
                           BiquadParameter(name="Biquad 1", removable=True, renamable=True)])
t = ParameterTree()
t.setParameters(p, showTop=True)

plot = pg.PlotWidget()
plot.setXRange(20, SAMPLERATE/2)
plot.showGrid(x=True, y=True)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t)
layout.addWidget(plot)

win.show()
win.resize(800, 800)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
