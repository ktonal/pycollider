#import pycollider
from ._pycollider import Unit, get_buffer_length, find_ugen, register_buffer
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

    def composeUnaryOp(self, aSelector):
        return UnaryOpUGen.new(aSelector, self)

    def neg(self):
        return self.composeUnaryOp('neg')

    def reciprocal(self):
        return self.composeUnaryOp('reciprocal')

    def bitNot(self):
        return self.composeUnaryOp('bitNot')

    def abs(self):
        return self.composeUnaryOp('abs')

    def asFloat(self):
        return self.composeUnaryOp('asFloat')

    def asInteger(self):
        return self.composeUnaryOp('asInteger')

    def ceil(self):
        return self.composeUnaryOp('ceil')

    def floor(self):
        return self.composeUnaryOp('floor')

    def frac(self):
        return self.composeUnaryOp('frac')

    def sign(self):
        return self.composeUnaryOp('sign')

    def squared(self):
        return self.composeUnaryOp('squared')

    def cubed(self):
        return self.composeUnaryOp('cubed')

    def sqrt(self):
        return self.composeUnaryOp('sqrt')

    def exp(self):
        return self.composeUnaryOp('exp')

    def midicps(self):
        return self.composeUnaryOp('midicps')

    def cpsmidi(self):
        return self.composeUnaryOp('cpsmidi')

    def midiratio(self):
        return self.composeUnaryOp('midiratio')

    def ratiomidi(self):
        return self.composeUnaryOp('ratiomidi')

    def ampdb(self):
        return self.composeUnaryOp('ampdb')

    def dbamp(self):
        return self.composeUnaryOp('dbamp')

    def octcps(self):
        return self.composeUnaryOp('octcps')

    def cpsoct(self):
        return self.composeUnaryOp('cpsoct')

    def log(self):
        return self.composeUnaryOp('log')

    def log2(self):
        return self.composeUnaryOp('log2')

    def log10(self):
        return self.composeUnaryOp('log10')

    def sin(self):
        return self.composeUnaryOp('sin')

    def cos(self):
        return self.composeUnaryOp('cos')

    def tan(self):
        return self.composeUnaryOp('tan')

    def asin(self):
        return self.composeUnaryOp('asin')

    def acos(self):
        return self.composeUnaryOp('acos')

    def atan(self):
        return self.composeUnaryOp('atan')

    def sinh(self):
        return self.composeUnaryOp('sinh')

    def cosh(self):
        return self.composeUnaryOp('cosh')

    def tanh(self):
        return self.composeUnaryOp('tanh')

    def rand(self):
        return self.composeUnaryOp('rand')

    def rand2(self):
        return self.composeUnaryOp('rand2')

    def linrand(self):
        return self.composeUnaryOp('linrand')

    def bilinrand(self):
        return self.composeUnaryOp('bilinrand')
    
    def sum3rand(self):
        return self.composeUnaryOp('sum3rand')

    def distort(self):
        return self.composeUnaryOp('distort')
        
    def softclip(self):
        return self.composeUnaryOp('softclip')
        
    def coin(self):
        return self.composeUnaryOp('coin')
    
    def aeven(self):
        return self.composeUnaryOp('even')
    
    def rectWindow(self):
        return self.composeUnaryOp('rectWindow')

    def hanWindow(self):
        return self.composeUnaryOp('hanWindow')

    def welWindow(self):
        return self.composeUnaryOp('welWindow')

    def triWindow(self):
        return self.composeUnaryOp('triWindow')
    
    def scurve(self):
        return self.composeUnaryOp('scurve')
        
    def ramp(self):
        return self.composeUnaryOp('ramp')

    def degrad(self):
        return self * 0.01745329251994329547
        
    def raddeg(self):
        return self * 57.29577951308232286465
    
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

    def __lt__(self, op):
        return self.composeBinaryOp('<', op)
    
    def __le__(self, op):
        return self.composeBinaryOp('<=', op)
    
    def __gt__(self, op):
        return self.composeBinaryOp('>', op)
    
    def __ge__(self, op):
        return self.composeBinaryOp('>=', op)

    def round(self, op):
        return self.composeBinaryOp('round', op)

    def round(self, op):
        return self.composeBinaryOp('roundUp', op)

    def trunc(self, op):
        return self.composeBinaryOp('trunc', op)

    def atan2(self, op):
        return self.composeBinaryOp('atan2', op)

    def hypot(self, op):
        return self.composeBinaryOp('hypot', op)

    def difsqr(self, op):
        return self.composeBinaryOp('difsqr', op)

    def sumsqr(self, op):
        return self.composeBinaryOp('sumsqr', op)

    def sqrsum(self, op):
        return self.composeBinaryOp('sqrsum', op)

    def sqrdif(self, op):
        return self.composeBinaryOp('sqrdif', op)

    def absdif(self, op):
        return self.composeBinaryOp('absdif', op)

    def thresh(self, op):
        return self.composeBinaryOp('thresh', op)

    def amclip(self, op):
        return self.composeBinaryOp('amclip', op)

    def scaleneg(self, op):
        return self.composeBinaryOp('scaleneg', op)

    def clip2(self, op):
        return self.composeBinaryOp('clip2', op)

    def fold2(self, op):
        return self.composeBinaryOp('fold2', op)

    def wrap2(self, op):
        return self.composeBinaryOp('wrap2', op)

    def rrand(self, op):
        return self.composeBinaryOp('rrand', op)

    def exprand(self, op):
        return self.composeBinaryOp('exprand', op)

    def clip(self, lo=0.0, hi=1.0):
        return Clip.perform(Clip.methodSelectorForRate(rate), self, lo, hi)

    def wrap(self, lo=0.0, hi=1.0):
        return Wrap.perform(Clip.methodSelectorForRate(rate), self, lo, hi)

    def fold(self, lo=0.0, hi=1.0):
        return Fold.perform(Clip.methodSelectorForRate(rate), self, lo, hi)

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

    def blend(other, blendFrac=0.5):
        pan = blendFrac.linlin(0.0, 1.0, -1, 1)
        if rate == 'audio':
                return XFade2.ar(this, that, pan)
        if other.rate == 'audio':
                return XFade2.ar(that, this, pan.neg)
        return LinXFade2.perform(LinXFade2.methodSelectorForRate(rate), this, that, pan)

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
        if self.num_outputs > 1:
            res = np.zeros((self.num_outputs, num_samples), dtype=np.float32)
            for k in range(num_samples // self.buffer_length):
                for item in self.stack:
                    item.calc(64)
                for i in range(self.num_outputs):
                    res[i, k*64:k*64+64] = self.output_buffers[i][:]
        else:
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
        self.num_outputs = len(units)
        self.buffer_length = get_buffer_length()
        self.output_buffers = [x.output_buffers[0] for x in units]
        
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
        buffer_length = get_buffer_length()
        self.buffer_length = buffer_length
        self.output_buffer = np.zeros(buffer_length, dtype=np.float32)
        if array.dtype != np.float32:
            array = array.astype(np.float32)
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


class UGen(Unit, UGenOps):

    @classmethod
    def multi_new(cls, *args):
        print("multi", args)
        size = 0
        for item in args:
            if isinstance(item, (list, tuple)):
                size = max(size, len(item))
            elif isinstance(item, (MultiOutUGen, MultiUGen)):
                size = max(size, item.num_outputs)
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
        self.ugen_id = find_ugen(type(self).__name__)
        self.buffer_length = get_buffer_length()
        self.special_index = 0
        self.init(num_outputs, inputs)
        self.handle_outputs(num_outputs)
        self.handle_inputs(inputs)
        Unit.__init__(self, self.ugen_id, self.input_buffers, self.output_buffers, self.input_rates,
                                 self.output_rates, self.special_index)
        return self

    def handle_outputs(self, num_outputs):
        self.num_outputs = num_outputs
        if num_outputs > 1:
            self.channels = [OutputProxy(self.rate, self, i) for i in range(num_outputs)]
        self.output_buffers = [np.zeros(self.buffer_length, dtype=np.float32) for _ in range(num_outputs)]
        self.output_rates = ["a" for _ in range(num_outputs)]

    def handle_inputs(self, inputs):
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
                raise ValueError("UGen inputs have to be numpy.ndarray, UGen, float or int: got {}".format(input))


class MultiOutUGen(UGen):

    def __getitem__(self, i):
        return self.channels[i]


 
# OutputProxys are needed so that the outputs from  MultiOutUGens can be individually addressed.
class OutputProxy(UGenOps):
    __slots__ = ('source', 'index', 'rate', 'num_outputs')

    def __init__(self, rate, source,  index):
        self.source = source
        self.index = index
        self.rate = rate
        self.num_outputs = 1

    def calc(self, n):
        self.source.calc(n)

    @property
    def scheduled(self):
        return self.source.scheduled

    @scheduled.setter
    def scheduled(self, val):
        self.source.scheduled = val


class BasicOpUGen(UGen):
    operatorIndices = {'neg': 0,
                       '+': 0,
                       '-': 1,
                       '*': 2,
                       'bitNot': 4,
                       '/': 4,
                       'mod': 5,
                       'abs': 5,
                       'asFloat': 6,
                       'asInteger': 7,
                       'ceil': 8,
                       '<': 8,
                       '>': 9,
                       'floor': 9,
                       'frac': 10,
                       '<=': 10,
                       'sign': 11,
                       '>=': 11,
                       'squared': 12,
                       'sqrt': 14,
                       'exp': 15,
                       'reciprocal': 16,
                       'midicps': 17,
                       'cpsmidi': 18,
                       'midiratio': 19,
                       'round': 19,
                       'ratiomidi': 20,
                       'roundUp': 20,
                       'trunc': 21,
                       'dbamp': 21,
                       'ampdb': 22,
                       'atan2': 22,
                       'octcps': 23,
                       'hypot': 23,
                       'cpsoct': 24,
                       'log': 25,
                       'pow': 25,
                       'log2': 26,
                       'log10': 27,
                       'sin': 28,
                       'cos': 29,
                       'tan': 30,
                       'asin': 31,
                       'acos': 32,
                       'atan': 33,
                       'sinh': 34,
                       'difsqr': 34,
                       'cosh': 35,
                       'sumsqr': 35,
                       'sqrsum': 36,
                       'tanh': 36,
                       'rand': 37,
                       'thresh': 39,
                       'sqrdif': 37,
                       'absdif': 38,
                       'rand2': 38,
                       'linrand': 39,
                       'amclip': 40,
                       'bilinrand': 40,
                       'scaleneg': 41,
                       'sum3rand': 41,
                       'distort': 42,
                       'clip2': 42,
                       'softclip': 43,
                       'coin': 44,
                       'fold2': 44,
                       'wrap2': 45,
                       'rrand': 47,
                       'exprand': 48,
                       'rectWindow': 48,
                       'hanWindow': 49,
                       'welWindow': 50,
                       'triWindow': 51,
                       'ramp': 52,
                       'scurve': 53}


    def unit_init(self, rate, operator, num_outputs, *inputs):
        self.rate = rate
        self.scheduled = False
        self.ugen_id = find_ugen(type(self).__name__)
        self.buffer_length = get_buffer_length()
        self.special_index = self.operatorIndices[operator]
        self.init(num_outputs, inputs)
        self.handle_outputs(num_outputs)
        self.handle_inputs(inputs)
        Unit.__init__(self, self.ugen_id, self.input_buffers, self.output_buffers, self.input_rates,
                                 self.output_rates, self.special_index)
        return self


class UnaryOpUGen(BasicOpUGen):

    def __new__(cls, operator, op1):
        return cls.multi_new('audio', operator, 1, op1)

    @classmethod
    def ar(cls, operator, op1):
        return cls.multi_new('audio', operator, 1, op1)

    @classmethod
    def kr(cls, operator, op1):
        return cls.multi_new('control', operator, 1, op1)


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


