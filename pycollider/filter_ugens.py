from .ugens import UGen, MultiOutUGen

class LeakDC(UGen):

    def __init__(cls, sig=0.0, coef=0.995, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)
        
    @classmethod
    def ar(cls, sig=0.0, coef=0.995, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)

    @classmethod
    def kr(cls, sig=0.0, pos=0.0, level=1.0):
        return cls.multi_new('control', sig, coef).madd(mul, add)


class OnePole(UGen):

    def __new__(cls, sig=0.0, coef=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, coef=0.5, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)

    @classmethod
    def kr(cls, sig=0.0, coef=0.5, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, coef).madd(mul, add)


class OneZero(UGen):
    pass


class FOS(UGen):

    def __new__(cls, sig=0.0, a0=0.0, a1=0.0, b1=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, a0, a1, b1).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, a0=0.0, a1=0.0, b1=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, a0, a1, b1).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, a0=0.0, a1=0.0, b1=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, a0, a1, b1).madd(mul, add)


class SOS(UGen):

    def __new__(cls, sig=0.0, a0=0.0, a1=0.0, a2=0.0,  b1=0.0, b2=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, a0, a1, a2, b1, b2).madd(mul, add)
    
    @classmethod
    def ar(cls, sig=0.0, a0=0.0, a1=0.0, a2=0.0,  b1=0.0, b2=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, a0, a1, a2, b1, b2).madd(mul, add)

    @classmethod
    def kr(cls, sig=0.0, a0=0.0, a1=0.0, a2=0.0,  b1=0.0, b2=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, a0, a1, a2, b1, b2).madd(mul, add)


class Integrator(UGen):

    def __new__(cls, sig, coef=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)
    
    @classmethod
    def ar(cls, sig, coef=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, coef).madd(mul, add)

    @classmethod
    def kr(cls, sig, coef=1.0, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, coef).madd(mul, add)
    
class Decay(UGen):

    def __new__(cls, sig=0.0, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, decayTime).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, decayTime).madd(mul, add)

    @classmethod
    def kr(cls, sig=0.0, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, decayTime).madd(mul, add)

    
class Decay2(UGen):

    def __new__(cls, sig=0.0, attackTime=0.01, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, attackTime, decayTime).madd(mul, add)
    
    @classmethod
    def ar(cls, sig=0.0, attackTime=0.01, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', sig, attackTime, decayTime).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, attackTime=0.01, decayTime=1.0, mul=1.0, add=0.0):
        return cls.multi_new('control', sig, attackTime, decayTime).madd(mul, add)


class RLPF(UGen):
    def __new__(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, rq).madd(mul, add)


class RHPF(UGen):
    def __new__(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, rq).madd(mul, add)


class BPF(UGen):
    def __new__(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, rq).madd(mul, add)


class BRF(UGen):
    def __new__(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, rq=1.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, rq).madd(mul, add)


class LPF(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class HPF(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class BPeakEQ(UGen):
    def __new__(cls, freq=100, rq=1.0, db=0.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq, db).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, rq=1.0, db=0.0, mul=1, add=0):
        return cls.multi_new('audio', 1, freq, rq, db).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, rq=1.0, db=0.0, mul=1, add=0):
        return cls.multi_new('control', 1, freq, rq, db).madd(mul, add)

    @classmethod
    def sc(cls, freq=1200.0, rq=1.0, db=0.0):
        sr  = SampleRate.ir
        a = pow(10, db/40)
        w0 = pi * 2 * freq * SampleDur.ir
        alpha = w0.sin * 0.5 * rq
        b0rz = (1 + (alpha / a)).reciprocal
        a0 = (1 + (alpha * a)) * b0rz
        a2 = (1 - (alpha * a)) * b0rz
        b1 = 2.0 * w0.cos * b0rz
        b2 = (1 - (alpha / a)) * b0rz.neg
        return [a0, b1.neg, a2, b1, b2];


class Lag(UGen):

    def __new__(cls, sig=0.0, lagTime=0.1, mul=1.0, add=0.0):
        if sig.rate == 'scalar' or lagTime == 0.0:
            return sig.madd(mul, add)
        else:
            return cls.multi_new('audio', sig, lagTime).madd(mul, add)

    @classmethod
    def ar(cls, sig=0.0, lagTime=0.1, mul=1.0, add=0.0):
        if sig.rate == 'scalar' or lagTime == 0.0:
            return sig.madd(mul, add)
        else:
            return cls.multi_new('audio', sig, lagTime).madd(mul, add)

    @classmethod
    def kr(cls, sig=0.0, lagTime=0.1, mul=1.0, add=0.0):
        if sig.rate == 'scalar' or lagTime == 0.0:
            return sig.madd(mul, add)
        else:
            return cls.multi_new('control', sig, lagTime).madd(mul, add)

class Lag2(Lag):
    pass
class Lag3(Lag):
    pass
class Ramp(Lag):
    pass


class MoogFF(UGen):

    def __new__(cls, sig, freq=100, gain=2, reset=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, freq, gain, reset).madd(mul, add)

    @classmethod
    def ar(cls, sig, freq=100, gain=2, reset=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, freq, gain, reset).madd(mul, add)
        

class MoogVCF(UGen):

    def __new__(cls, sig, fco, res, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, fco, res).madd(mul, add)

    @classmethod
    def ar(cls, sig, fco, res, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, fco, res).madd(mul, add)


class SVF(UGen):

    def __new__(cls, signal, cutoff=2200.0, res=0.1, lowpass=1.0, bandpass=0.0, highpass=0.0,
                notch=0.0, peak=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', 1, signal, cutoff, res,
                             lowpass, bandpass, highpass, notch, peak).madd(mul, add)

    @classmethod
    def ar(cls, signal, cutoff=2200.0, res=0.1, lowpass=1.0, bandpass=0.0, highpass=0.0,
           notch=0.0, peak=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', 1, signal, cutoff, res,
                             lowpass, bandpass, highpass, notch, peak).madd(mul, add)

    @classmethod
    def ar(cls, signal, cutoff=2200.0, res=0.1, lowpass=1.0, bandpass=0.0, highpass=0.0,
           notch=0.0, peak=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', 1, signal, cutoff, res,
                             lowpass, bandpass, highpass, notch, peak).madd(mul, add)

    
class OBXFilter(UGen):

    def __new__(cls, sig, freq=400, res=0.4, fourpole=0, mode=0, sattype=7,
                blend=0.9, noise=0.001, mul=1, add=0): 
        return cls.multi_new('audio', 1, sig, freq, res, fourpole, mode,
                             sattype, blend, noise).madd(mul, add)

    @classmethod
    def ar(cls, sig, freq=400, res=0.4, fourpole=0, mode=0, sattype=7,
                blend=0.9, noise=0.001, mul=1, add=0): 
        return cls.multi_new('audio', 1, sig, freq, res, fourpole, mode,
                             sattype, blend, noise).madd(mul, add)


class MLadder(UGen):

    def __new__(cls, sig, freq=100, gain=2, a0=0, a1=0, a2=0, a3=0, a4=1, st1=5, st2=7,
                st3=7, st4=7, sb1=0.9, sb2=0.9, sb3=0.9, sb4=0.0, noise=0.0002, chdel=200,
                chorus=0.3, automod=20, mul=1, add=0): 
        return self.multi_new('audio', 1, sig, freq, gain, a0, a1, a2, a3, a4,
                              st1, st2, st3, st4, sb1, sb2, sb3, sb4, noise,
                              chdel, chorus, automod).madd(mul, add)

    @classmethod
    def ar(cls, sig, freq=100, gain=2, a0=0, a1=0, a2=0, a3=0, a4=1, st1=5, st2=7,
                st3=7, st4=7, sb1=0.9, sb2=0.9, sb3=0.9, sb4=0.0, noise=0.0002, chdel=200,
                chorus=0.3, automod=20, mul=1, add=0): 
        return self.multi_new('audio', 1, sig, freq, gain, a0, a1, a2, a3, a4,
                              st1, st2, st3, st4, sb1, sb2, sb3, sb4, noise,
                              chdel, chorus, automod).madd(mul, add)


class MoogImproved(UGen):

    def __new__(cls, sig, freq=100, res=0.3, drive=1, noise=0.0001,
                c1=0,c2=0,c3=0,c4=0,c5=1, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, freq, res, drive, noise,
                             c1, c2, c3, c4, c5).madd(mul, add)

    @classmethod
    def ar(cls, sig, freq=100, res=0.3, drive=1, noise=0.0001,
                c1=0,c2=0,c3=0,c4=0,c5=1, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, freq, res, drive, noise,
                             c1, c2, c3, c4, c5).madd(mul, add)


class LPF18(UGen):

    def __new__(cls, sig, freq=100, res=1, dist=0.4):
        return cls.multi_new('audio', 1, sig, freq, res, dist)

    @classmethod
    def ar(cls, sig, freq=100, res=1, dist=0.4):
        return cls.multi_new('audio', 1, sig, freq, res, dist)


class Hilbert(UGen):

    def __new__(cls, sig, mul=1, add=0):
        return cls.multi_new('audio', 2, sig).madd(mul, add)

    @classmethod
    def ar(cls, sig, mul=1, add=0):
        return cls.multi_new('audio', 2, sig).madd(mul, add)
