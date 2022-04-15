import pycollider
import numpy as np
import sys
import gc
import pprint
import copy
from collections import defaultdict


class UGenOps:

    @property
    def signalRange(self):
        return 'bipolar'
        
    def composeBinaryOp(self, aSelector, anInput):
        return BinaryOpUGen(aSelector, self, anInput)

    def __add__(self, op):
        return self.composeBinaryOp('+', op)

    def __radd__(self, op):
        return self.composeBinaryOp('+', op)

    def __sub__(self, op):
        return self.composeBinaryOp('-', op)

    def __rsub__(self, op):
        return self.composeBinaryOp('-', op)

    def __mul__(self, op):
        return self.composeBinaryOp('*', op)

    def __rmul__(self, op):
        return self.composeBinaryOp('*', op)

    def __div__(self, op):
        return self.composeBinaryOp('/', op)

    def __rdiv__(self, op):
        return self.composeBinaryOp('/', op)

    def __mod__(self, op):
        return self.composeBinaryOp('mod', op)

    def __rmod__(self, op):
        return self.composeBinaryOp('mod', op)

    def __pow__(self, op):
        return self.composeBinaryOp('pow', op)

    def __rpow__(self, op):
        return self.composeBinaryOp('pow', op)

    def madd(self, mul=1, add=0):
        return MulAdd(self, mul, add)

    def range(self, lo = 0.0, hi = 1.0):
        if self.signalRange == 'bipolar':
            mul = (hi - lo) * 0.5
            add = mul + lo
        else:
            mul = (hi - lo)
            add = lo
        return MulAdd(self, mul, add)

    def collect_inputs_helper(self, unit):
        for input in unit.inputs:
            self.all_inputs.add(input)
            self.graph[input].append(unit)
            self.collect_inputs_helper(input)
        
    def collect_inputs(self):
        self.all_inputs = set()
        self.graph = defaultdict(list)
        self.collect_inputs_helper(self)
        self.all_inputs = list(self.all_inputs)

    def toposort_helper(self, v):
        # Recur for all the vertices adjacent to this vertex
        v.scheduled = True
        for i in self.graph[v]:
            if not i.scheduled:
                self.toposort_helper(i)
                # Push current vertex to stack which stores result
        self.stack.insert(0,v)
 
    def toposort(self):
        # Mark all the vertices as not visited
        self.stack =[]
        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in self.all_inputs:
            if not i.scheduled:
                self.toposort_helper(i)
        if self not in self.stack:
            self.stack.append(self)
        return self.stack

    def render(self, num_samples):
        self.collect_inputs()
        self.toposort()
        res = np.zeros(num_samples, dtype=np.float32)
        for k in range(num_samples // self.buffer_length):
            for item in self.stack:
                item.calc(64)
            res[k*64:k*64+64] = self.output_buffers[0][:]
        return res

    
class MultiUGen(UGenOps):

    def __init__(self, units):
        self.scheduled = False
        self.units = units

    def __getitem__(self, k):
        return self.units[k]

    def __setitem__(self, k, unit):
        self.units[k] = unit

    def __len__(self):
        return len(self.units)

    def calc(self, n):
        for unit in self.units:
            unit.calc(n)

    def madd(self, mul=1, add=0):
        self.units = [MulAdd(unit, mul, add) for unit in self.units]
        return self

    @property
    def inputs(self):
        res = []
        for unit in self.units:
            res += unit.inputs
        return res


class BufferFeeder:

    def __init__(self, array):
        self.scheduled = False
        buffer_length = pycollider.get_buffer_length()
        self.buffer_length = buffer_length
        self.output_buffer = np.zeros(buffer_length, dtype=np.float32)
        self.array = array
        self.idx = 0
        
    def calc(self, inNumSamples):
        self.output_buffer[:] = self.array[self.idx * self.buffer_length: (self.idx + 1) * self.buffer_length]
        self.idx += 1


class ScalarFeeder:

    def __init__(self, value):
        self.scheduled = True
        self.value = value
        
    def calc(self, inNumSamples):
        pass

    @property
    def inputs(self):
        return []
    
class UGen(pycollider.Unit, UGenOps):

    @classmethod
    def multi_new(cls, *args):
        print("multi", args)
        size = 0
        for item in args:
            if isinstance(item, (list, tuple)):
                size = max(size, len(item))
        if size == 0:
            # single unit
            return super().__new__(cls).unit_init(*args)
        new_args = [0] * len(args)
        results = [0] * size
        for i in range(size):
            for j, item in enumerate(args):
                if isinstance(item, (list, tuple)):
                    new_args[j] = item[i % len(item)]
                else:
                    new_args[j] = item
            results[i] = cls.multi_new(*new_args)
        return MultiUGen(results)

    def __init__(self, *args):
        print("UGen.__init__", self, args)

    def init(self, *args):
        pass
    
    def unit_init(self, rate, num_outputs, *inputs):
        print("init", rate, num_outputs, inputs)
        self.rate = rate
        self.scheduled = False
        self.ugen_id = pycollider.find_ugen(type(self).__name__)
        self.buffer_length = pycollider.get_buffer_length()
        self.special_index = 0
        self.handle_outputs(num_outputs)
        self.handle_inputs(inputs)
        self.init(num_outputs, inputs)
        pycollider.Unit.__init__(self, self.ugen_id, self.input_buffers, self.output_buffers, self.input_rates,
                                 self.output_rates, self.special_index)
        return self

    def handle_outputs(self, num_outputs):
        self.num_outputs = num_outputs
        self.output_buffers = [np.zeros(self.buffer_length, dtype=np.float32) for _ in range(num_outputs)]
        self.output_rates = ["a" for _ in range(num_outputs)]

    def handle_inputs(self, inputs):
        self.inputs = inputs
        self.input_buffers = []
        self.input_rates = []
        self.inputs = []
        for input in inputs:
            if isinstance(input, np.ndarray):
                bf = BufferFeeder(input)
                self.inputs += [bf]
                self.input_buffers += [bf.output_buffer]
                self.input_rates += ["a"]
            elif isinstance(input, (int, float)):
                self.inputs += [ScalarFeeder(input)]
                self.input_buffers += [np.array([float(input)], dtype=np.float32)]
                self.input_rates += ["k"]
            elif isinstance(input, UGen):
                self.inputs += [input]
                if input.rate == 'audio':                    
                    self.input_rates += ["a"]
                    self.input_buffers += [input.output_buffers[0]]
                else:                    
                    self.input_rates += ["k"]
                    self.input_buffers += [input.output_buffers[0]]
            else:
                raise ValueError("UGen inputs have to be numpy.ndarray, UGen, float or int")



class BasicOpUGen(UGen):
    operatorIndices = {'+': 0, '-': 1, '*': 2, '/': 4, 'mod': 5, 'pow': 25}

    def unit_init(self, rate, operator, num_outputs, *inputs):
        self.rate = rate
        self.scheduled = False
        self.ugen_id = pycollider.find_ugen(type(self).__name__)
        self.buffer_length = pycollider.get_buffer_length()
        self.special_index = self.operatorIndices[operator]
        self.handle_outputs(num_outputs)
        self.handle_inputs(inputs)
        self.init(num_outputs, inputs)
        pycollider.Unit.__init__(self, self.ugen_id, self.input_buffers, self.output_buffers, self.input_rates,
                                 self.output_rates, self.special_index)
        return self



class BinaryOpUGen(BasicOpUGen):
    def __new__(cls, operator, op1, op2):
        return cls.multi_new('audio', operator, 1, op1, op2)

    @classmethod
    def ar(cls, operator, op1, op2):
        return cls.multi_new('audio', operator, 1, op1, op2)

    @classmethod
    def kr(cls, operator, op1, op2):
        return cls.multi_new('control', operator, 1, op1, op2)


def rate(item):
    if isinstance(item, (float, int)):
        return 'scalar'
    else:
        return item.rate


class MulAdd(UGen):

    def __new__(cls, inp, mul = 1, add = 0):
        #eliminate degenerate cases
        if mul == 0.0:
            return add
        minus = mul == -1.0
        nomul = mul == 1.0
        noadd = add == 0.0
        if nomul and noadd:
            return inp
        if minus and noadd:
            return inp.neg
        if noadd:
            return inp * mul
        if minus:
            return add - inp
        if nomul:
            return inp + add

        if MulAdd.canBeMulAdd(inp, mul, add):
            return cls.multi_new('audio', 1, inp, mul, add)

        if MulAdd.canBeMulAdd(mul, inp, add):
            return cls.multi_new('audio', 1, mul, inp, add)

        return inp * mul + add

    @staticmethod
    def canBeMulAdd(inp, mul, add):
        # see if these inputs satisfy the constraints of a MulAdd ugen.
        if rate(inp) == 'audio':
            return True
        if rate(inp) == 'control' and  (rate(mul) == 'control' or  rate(mul) == 'scalar') \
           and (rate(add) == 'control' or rate(add) == 'scalar'):
            return True
        return False



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
    def sc(cls, freq = 1200.0, rq = 1.0, db = 0.0):
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


class PlayBuf(UGen):
    def __new__ (cls, numChannels, bufnum=0, rate=1.0, trigger=1.0, startPos=0.0, loop = 0.0, doneAction=0):
        return cls.multi_new('audio', numChannels, bufnum, rate, trigger, startPos, loop, doneAction)

    def init(self, argNumChannels, *inputs):
        self.num_outputs = argNumChannels

