from .ugens import UGen, MultiOutUGen

class WhiteNoise(UGen):
    def __new__(cls,  mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def ar(cls, mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def kr(cls,  mul=1, add=0):
        return cls.multi_new('control', 1).madd(mul, add)


class PinkNoise(UGen):
    def __new__(cls,  mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def ar(cls, mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def kr(cls,  mul=1, add=0):
        return cls.multi_new('control', 1).madd(mul, add)


class BrownNoise(UGen):
    def __new__(cls,  mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def ar(cls, mul=1, add=0):
        return cls.multi_new('audio', 1).madd(mul, add)

    @classmethod
    def kr(cls,  mul=1, add=0):
        return cls.multi_new('control', 1).madd(mul, add)


class LFNoise0(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFNoise1(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFNoise2(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFClipNoise(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFDNoise1(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFDNoise0(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFDNoise3(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


class LFCDlipNoise(UGen):
    def __new__(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def ar(cls, freq=100, mul=1, add=0):
        return cls.multi_new('audio', 1, freq).madd(mul, add)

    @classmethod
    def kr(cls, freq=100, mul=1, add=0):
        return cls.multi_new('control', 1, freq).madd(mul, add)


