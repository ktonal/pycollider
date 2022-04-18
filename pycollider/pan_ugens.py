from .ugens import UGen, MultiOutUGen

class XFade2(UGen):

    def __new__(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('audio', inA, inB, pan, level)

    @classmethod
    def ar(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('audio', inA, inB, pan, level)

    @classmethod
    def kr(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('control', inA, inB, pan, level)

    def checkInputs(self):
        return self.checkNInputs(2)


class LinXFade2(UGen):

    def __new__(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('audio', inA, inB, pan) * level

    @classmethod
    def ar(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('audio', inA, inB, pan) * level

    @classmethod
    def kr(cls, inA, inB=0.0, pan=0.0, level=1.0):
        return cls.multi_new('control', inA, inB, pan) * level

    def checkInputs(self):
        return self.checkNInputs(2)


class Pan2(MultiOutUGen):

    def __new__(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('audio', 2, sig, pos, level)

    @classmethod
    def ar(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('audio', 2, sig, pos, level)

    @classmethod
    def kr(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('control', 2, sig, pos, level)

    def checkInputs(self):
        return self.checkNInputs(1)


class LinPan2(Pan2):

    def __new__(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('audio', sig, pos, level)


class Pan2(MultiOutUGen):

    def __new__(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('audio', 2, sig, pos, level)

    @classmethod
    def ar(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('audio', 2, sig, pos, level)

    @classmethod
    def kr(cls, sig, pos=0.0, level=1.0):
        return cls.multi_new('control', 2, sig, pos, level)

    def checkInputs(self):
        return self.checkNInputs(1)


class Balance2(MultiOutUGen):

    def __new__(cls, left, right, pos=0.0, level=1.0):
        return cls.multiNew('audio', left, right, pos, level)

    @classmethod
    def ar(cls, left, right, pos=0.0, level=1.0):
        return cls.multiNew('audio', left, right, pos, level)

    @classmethod
    def kr(cls, left, right, pos=0.0, level=1.0):
        return cls.multiNew('control', left, right, pos, level)

    def init(self, *theInputs):
        self.inputs = theInputs
        self.channels = [OutputProxy(rate, self, 0), OutputProxy(rate, self, 1)]
        return channels
        
    def checkInputs(self):
        return self.checkNInputs(2)



