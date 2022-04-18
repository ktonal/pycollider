from .ugens import UGen, MultiOutUGen


class SatAmp(UGen):

    def __new__(cls, sig, pregain=1, sattype=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, pregain, sattype).madd(mul, add)

    @classmethod
    def ar(cls, sig, pregain=1, sattype=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, pregain, sattype).madd(mul, add)

    @classmethod
    def kr(cls, sig, pregain=1, sattype=0, mul=1, add=0):
        return cls.multi_new('control', 1, sig, pregain, sattype).madd(mul, add)


class SatAmp4(UGen):

    def __new__(cls, sig, pregain=1, sattype=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, pregain, sattype).madd(mul, add)

    @classmethod
    def ar(cls, sig, pregain=1, sattype=0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, pregain, sattype).madd(mul, add)


class HardClip2(UGen):

    def __new__(cls, sig, lower=-1.0, upper=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, lower, upper).madd(mul, add)

    @classmethod
    def ar(cls, sig, lower=-1.0, upper=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, lower, upper).madd(mul, add)


class HardClip(UGen):

    def __new__(cls, sig, lower=-1.0, upper=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, lower, upper).madd(mul, add)

    @classmethod
    def ar(cls, sig, lower=-1.0, upper=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, lower, upper).madd(mul, add)


class WaveFolder(UGen):

    def __new__(cls, sig, anti_alias=1, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, anti_alias).madd(mul, add)

    @classmethod
    def ar(cls, sig, lower=-1.0, upper=1.0, mul=1, add=0):
        return cls.multi_new('audio', 1, sig, lower, upper).madd(mul, add)


class HardSoftClip2(UGen):

    def __new__(cls, sig, soft=0.9, blend=0.9, c1=0, c2=0, c3=1, c4=0, c5=0, mul=0.3, add=0):
        return self.multi_new('audio', 1, sig, soft, blend, c1, c2, c3, c4, c5).madd(mul, add)

    @classmethod
    def ar(cls, sig, soft=0.9, blend=0.9, c1=0, c2=0, c3=1, c4=0, c5=0, mul=0.3, add=0):
        return self.multi_new('audio', 1, sig, soft, blend, c1, c2, c3, c4, c5).madd(mul, add)
    

class FreqShift(UGen):

    def __new__(cls, sig, 
                freq=0.0,       # shift, in cps
                phase=0.0,      # phase of SSB
                mul=1.0,
                add=0.0):
        return cls.multi_new('audio', 2, sig, freq, phase).madd(mul, add)

    @classmethod
    def ar(cls, sig, 
           freq=0.0,       # shift, in cps
           phase=0.0,      # phase of SSB
           mul=1.0,
           add=0.0):
        return cls.multi_new('audio', 2, sig, freq, phase).madd(mul, add)

    @classmethod
    def kr(cls, sig, 
           freq=0.0,       # shift, in cps
           phase=0.0,      # phase of SSB
           mul=1.0,
           add=0.0):
        return cls.multi_new('control', 2, sig, freq, phase).madd(mul, add)


class DiodeRingMod(UGen):

    def __new__(cls, car=0.0, mod=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', car, mod).madd(mul, add)

    @classmethod
    def ar(cls, car=0.0, mod=0.0, mul=1.0, add=0.0):
        return cls.multi_new('audio', car, mod).madd(mul, add)

    @classmethod
    def kr(cls, car=0.0, mod=0.0, mul=1.0, add=0.0):
        return cls.multi_new('control', car, mod).madd(mul, add)
