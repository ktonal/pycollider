from .ugens import UGen, MultiOutUGen

class SinOsc(UGen):
    def __new__(cls, freq=200, phs=0.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs).madd(mul, add)

    @classmethod
    def ar(cls, freq=200, phs=0.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs).madd(mul, add)

    @classmethod
    def kr(cls, freq=200, phs=0.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, phs).madd(mul, add)


class Saw(UGen):
    def __new__(cls, freq=200, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=200, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=200, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class Pulse(UGen):
    def __new__(cls, freq=200, width=0.5, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, width).madd(mul, add)

    @classmethod
    def ar(cls, freq=200, width=0.5, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, width).madd(mul, add)

    @classmethod
    def kr(cls, freq=200, width=0.5, mul=1, add=0):
        return cls.multi_new('control', 1, freq, width).madd(mul, add)



class SawPTR(UGen):
    def __new__(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs, trig).madd(mul, add)

    @classmethod
    def ar(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs, trig).madd(mul, add)

    @classmethod
    def kr(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, phs, trig).madd(mul, add)


class PulsePTR(UGen):
    def __new__(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs, trig).madd(mul, add)

    @classmethod
    def ar(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, phs, trig).madd(mul, add)

    @classmethod
    def kr(cls, freq=200, phs=0.0, trig=-1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, phs, trig).madd(mul, add)


class PlayBuf(UGen):
    def __new__ (cls, numChannels, bufnum=0, rate=1.0, trigger=1.0,
                 startPos=0.0, loop = 0.0, doneAction=0):
        return cls.multi_new('audio', numChannels, bufnum, rate, trigger,
                             startPos, loop, doneAction)

    @classmethod
    def ar(cls, numChannels, bufnum=0, rate=1.0, trigger=1.0,
           startPos=0.0, loop = 0.0, doneAction=0):
        return cls.multi_new('audio', numChannels, bufnum, rate, trigger,
                             startPos, loop, doneAction)

    @classmethod
    def kr(cls, numChannels, bufnum=0, rate=1.0, trigger=1.0,
           startPos=0.0, loop = 0.0, doneAction=0):
        return cls.multi_new('control', numChannels, bufnum, rate, trigger,
                             startPos, loop, doneAction)

    def init(self, argNumChannels, inputs):
        if isinstance(inputs[0], np.ndarray):
            bufnum = register_buffer(-1, inputs[0])
            inputs[0] = bufnum
        print("playbuf init", inputs)
        self.num_outputs = argNumChannels
        self.inputs = inputs


class Blip(UGen):

    def __new__(cls, freq=440.0, numharm = 200.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, numharm).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, numharm = 200.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, numharm).madd(mul, add)

    @classmethod
    def kr(cls, freq=440.0, numharm = 200.0, mul=1.0, add=0.0):
        return cls.multi_new('control', freq, numharm).madd(mul, add)


class Formant(UGen):

    def __new__(cls, fundfreq=440.0, formfreq=1760, bwfreq=800, mul=1.0, add=0.0):
        return cls.multi_new('audio', fundfreq, formfreq, bwfreq).madd(mul, add)

    @classmethod
    def ar(cls, fundfreq=440.0, formfreq=1760, bwfreq=800, mul = 1.0, add = 0.0):
        return cls.multi_new('audio', fundfreq, formfreq, bwfreq).madd(mul, add)


class LFSaw(UGen):

    def __new__(cls, freq=440.0, iphase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, iphase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase).madd(mul, add)

    @classmethod
    def kr(cls, freq=440.0, iphase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', freq, iphase).madd(mul, add)



class LFPar(LFSaw):
    pass
class LFCub(LFSaw):
    pass
class LFTri(LFSaw):
    pass


class LFGauss(UGen):

    def __new__(cls, duration=1, width=0.1, iphase=0.0, loop=1, doneAction=0):
        return cls.multi_new('audio', duration, width, iphase, loop, doneAction)

    @classmethod
    def __new__(cls, duration=1, width=0.1, iphase=0.0, loop=1, doneAction=0):
        return cls.multi_new('audio', duration, width, iphase, loop, doneAction)

    @classmethod
    def __new__(cls, duration=1, width=0.1, iphase=0.0, loop=1, doneAction=0):
        return cls.multi_new('control', duration, width, iphase, loop, doneAction)

    def range(self, min=0, max=1):
        return self.linlin(this.minval, 1, min, max)
        
    def minval(self):
        width = self.inputs[1];
        return (1.0 / (-2.0 * width * width)).exp()

    
class LFPulse(UGen):

    def __new__(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase, width).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase, width).madd(mul, add)

    @classmethod
    def kr(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('control', freq, iphase, width).madd(mul, add)

    @property
    def signalRange(self):
        return 'unipolar'


class VarSaw(UGen):

    def __new__(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase, width).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, iphase, width).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, iphase=0.0, width=0.5, mul=1.0, add=0.0):
        return cls.multi_new('control', freq, iphase, width).madd(mul, add)


class Impulse(UGen):

    def __new__(cls, freq=440.0, phase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, phase).madd(mul, add)

    @classmethod
    def ar(cls, freq=440.0, phase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', freq, phase).madd(mul, add)

    @classmethod
    def kr(cls, freq=440.0, phase=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', freq, phase).madd(mul, add)

    @property
    def signalRange(self):
        return 'unipolar'


class SyncSaw(UGen):

    def __new__(cls, syncFreq=440.0, sawFreq=440,  mul=1.0, add=0.0):
        return cls.multi_new('audio', syncFreq, sawFreq).madd(mul, add)

    @classmethod
    def ar(cls, syncFreq=440.0, sawFreq=440,  mul=1.0, add=0.0):
        return cls.multi_new('audio', syncFreq, sawFreq).madd(mul, add)

    @classmethod
    def kr(cls, syncFreq=440.0, sawFreq=440,  mul=1.0, add=0.0):
        return cls.multi_new('control', syncFreq, sawFreq).madd(mul, add)

