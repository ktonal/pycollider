from .ugens import UGen, MultiOutUGen


class DelayN(UGen):

    def __new__(cls, sig, maxdelaytime = 0.2, delaytime = 0.2, mul = 1.0, add = 0.0):
        return cls.multi_new('audio', 1, sig, maxdelaytime, delaytime).madd(mul, add)

    @classmethod
    def ar(cls, sig, maxdelaytime = 0.2, delaytime = 0.2, mul = 1.0, add = 0.0):
        return cls.multi_new('audio', 1, sig, maxdelaytime, delaytime).madd(mul, add)

    @classmethod
    def kr(cls, sig, maxdelaytime = 0.2, delaytime = 0.2, mul = 1.0, add = 0.0):
        return cls.multi_new('control', 1, sig, maxdelaytime, delaytime).madd(mul, add)
    
class DelayL(DelayN):
    pass

class DelayC(DelayN):
    pass


class CombN(UGen):

    def __new__(cls, sig, maxdelaytime=0.2, delaytime=0.2, decaytime=1.0, mul=1.0, add=0.0):
        return cls.multi_ew('audio', 1, sig, maxdelaytime, delaytime, decaytime).madd(mul, add)

    @classmethod
    def ar(cls, sig, maxdelaytime=0.2, delaytime=0.2, decaytime=1.0, mul=1.0, add=0.0):
        return cls.multi_ew('audio', 1, sig, maxdelaytime, delaytime, decaytime).madd(mul, add)

    @classmethod
    def kr(cls, sig, maxdelaytime=0.2, delaytime=0.2, decaytime=1.0, mul=1.0, add=0.0):
        return cls.multi_ew('control', 1, sig, maxdelaytime, delaytime, decaytime).madd(mul, add)


class CombL(CombN):
    pass

class CombC(CombN):
    pass

class AllpassN(CombN):
    pass

class AllpassL(CombN):
    pass

class AllpassC(CombN):
    pass

