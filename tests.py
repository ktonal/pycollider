import pycollider
import numpy as np
import matplotlib.pyplot as plt
import numbers
import soundfile

plt.ion()

plugins = ["/home/berlach/work/repos/berlach-plugins/build/cpp/BinaryOpUGens.so",
"/usr/local/lib/SuperCollider/plugins/ChaosUGens.so",
"/home/berlach/work/repos/berlach-plugins/build/cpp/DelayUGens.so",
"/usr/local/lib/SuperCollider/plugins/DynNoiseUGens.so",
"/usr/local/lib/SuperCollider/plugins/FilterUGens.so",
"/usr/local/lib/SuperCollider/plugins/GendynUGens.so",
"/usr/local/lib/SuperCollider/plugins/GrainUGens.so",
"/usr/local/lib/SuperCollider/plugins/IOUGens.so",
"/usr/local/lib/SuperCollider/plugins/LFUGens.so",
"/usr/local/lib/SuperCollider/plugins/ML_UGens.so",
"/home/berlach/work/repos/berlach-plugins/build/cpp/MulAddUGens.so",
"/usr/local/lib/SuperCollider/plugins/NoiseUGens.so",
"/usr/local/lib/SuperCollider/plugins/OscUGens.so",
"/usr/local/lib/SuperCollider/plugins/PanUGens.so",
"/usr/local/lib/SuperCollider/plugins/PhysicalModelingUGens.so",
"/usr/local/lib/SuperCollider/plugins/PV_ThirdParty.so",
"/usr/local/lib/SuperCollider/plugins/ReverbUGens.so",
"/usr/local/lib/SuperCollider/plugins/TriggerUGens.so",
"/usr/local/lib/SuperCollider/plugins/UnaryOpUGens.so",
"/usr/local/lib/SuperCollider/plugins/FFT_UGens.so",
"/usr/local/lib/SuperCollider/plugins/DemandUGens.so",
"/home/berlach/.local/share/SuperCollider/Extensions/BerlachFilters.so",
"/home/berlach/.local/share/SuperCollider/Extensions/BerlachUGens.so",
"/home/berlach/.local/share/SuperCollider/Extensions/Compressors.so",
"/home/berlach/.local/share/SuperCollider/Extensions/FirstOrderFB.so",
"/home/berlach/.local/share/SuperCollider/Extensions/OSNonlinear.so",
"/home/berlach/.local/share/SuperCollider/Extensions/HystPlayOp.so",
"/home/berlach/.local/share/SuperCollider/Extensions/PTROsc.so",
"/home/berlach/.local/share/SuperCollider/Extensions/ReverbUGens.so"]


for plug in plugins:
    pycollider.load_plugin(plug)

from pycollider.ugens import *

unit1 = SinOsc(800)
sig = unit1.render(44100) * np.linspace(1.0,  0.0, 44100, dtype=np.float32)

plt.plot(sig)

sig = np.array([sig])

pycollider.register_buffer(0, sig)
sig2 = PlayBuf(1, 0)

sig2.input_rates

outsig = sig2.render(22000)


unit1 = ((Saw(200) * 200) + 100).render(44100)


unit1 = Saw(200).madd(200, 100)


unit1 = Saw(Saw(44).range(20, 1400), 0.3)





unit1.collect_inputs()
unit1.toposort()

res = np.zeros(64*101, dtype=np.float32)
for k in range(100):
    for item in unit1.stack:
        item.calc(64)
    res[k*64:k*64+64] = unit1.output_buffers[0][:]

plt.plot(res)

    
outbuf = np.zeros(128, dtype=np.float32)
freq = np.ones(128, dtype=np.float32) * 400
phs = np.zeros(64*21, dtype=np.float32)
sync = np.ones(64*21, dtype=np.float32) * -1

unit = SinOsc.ar(np.linspace(200, 1900, 64*21), phs)

unit1 = SinOsc.ar(500, 0)
unit2 = SinOsc.ar(800, 0)

res = np.zeros(64*101, dtype=np.float32)





calc_queue = []





units = list(units)


def topologicalSortUtil(v, stack):
     # Recur for all the vertices adjacent to this vertex
    v.scheduled = True
    for i in graph[v]:
        if not i.scheduled:
            topologicalSortUtil(i, stack)
    # Push current vertex to stack which stores result
    stack.insert(0,v)
 
# The function to do Topological Sort. It uses recursive
# topologicalSortUtil()
def topologicalSort():
    # Mark all the vertices as not visited
    stack =[]
    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for i in units:
        if not i.scheduled:
            topologicalSortUtil(i, stack)
    # Print contents of stack
    return stack


schedule = topologicalSort()

units = set()
def collect_inputs(unit):
    global units
    for input in unit.inputs:
        units.add(input)
        graph[input].append(unit)
        collect_inputs(input)

unit1 = 0.7 * Saw(Saw(44).range(100, 500), 0.4)

unit1.collect_inputs()
unit1.toposort()

unit1 = Saw(231)


unit1 = 0


units = set()
graph = defaultdict(list) #dictionary containing adjacency List
collect_inputs(unit1)    
units = list(units)

opunit = SinOsc.ar(800, 0, 0.4) + SinOsc.ar(440, 0, 0.7)

opunit = SinOsc.ar(800, 0) + SinOsc.ar(440, 0)

unit = Saw(200, 0.2)

for k in range(20):
    unit.calc(64)
    res[k*64:k*64+64] = unit.output_buffers[0][:]

plt.plot(res)

res = np.zeros(64*101, dtype=np.float32)
for k in range(20):
    for input in unit.inputs:
        input.calc(64)
    unit.calc(64)
    res[k*64:k*64+64] = unit.output_buffers[0][:]

plt.ion()

unit = 0

plt.plot(res)

opunit = BinaryOpUGen.ar('+', unit1, unit2)

res = np.zeros(64*101, dtype=np.float32)


opunit = MulAdd.init(unit1, unit2, 0.5)


for k in range(20):
    unit1.calc(64)
    unit2.calc(64)
    opunit.calc(64)
    res[k*64:k*64+64] = opunit.output_buffers[0][:]


class MulAdd(UGen):

    @classmethod
    def init(self, in, mul, add):
        #eliminate degenerate cases
        if mul == 0.0:
            return add
        minus = mul == -1.0;
        nomul = mul == 1.0;
        noadd = add == 0.0;
        if nomul and noadd:
            return in
        if minus and noadd:
            return in.neg
        if noadd:
            return in * mul
        if minus:
            return add - in
        if nomul:
            return in + add

        if self.canBeMulAdd(in, mul, add): 
             super.new1(rate, in, mul, add)

        if self.canBeMulAdd(mul, in, add):
            super.new1(rate, mul, in, add)

         ^( (in * mul) + add)




	var <>antecedents, <>descendants, <>widthFirstAntecedents; // topo sorting

        	*multiNewList { arg args;
		var size = 0, newArgs, results;
		args = args.asUGenInput(this);
		args.do({ arg item;
			(item.class == Array).if({ size = max(size, item.size) });
		});
		if (size == 0) { ^this.new1( *args ) };
		newArgs = Array.newClear(args.size);
		results = Array.newClear(size);
		size.do({ arg i;
			args.do({ arg item, j;
				newArgs.put(j, if (item.class == Array, { item.wrapAt(i) },{ item }));
			});
			results.put(i, this.multiNewList(newArgs));
		});
		^results
	}

                	madd { arg mul = 1.0, add = 0.0;
		^MulAdd(this, mul, add);
	}
	range { arg lo = 0.0, hi = 1.0;
		var mul, add;
		if (this.signalRange == \bipolar, {
			mul = (hi - lo) * 0.5;
			add = mul + lo;
		},{
			mul = (hi - lo) ;
			add = lo;
		});
		^MulAdd(this, mul, add);
	}
	exprange { arg lo = 0.01, hi = 1.0;
		^if (this.signalRange == \bipolar) {
			this.linexp(-1, 1, lo, hi, nil)
		} {
			this.linexp(0, 1, lo, hi, nil)
		};
	}

	curverange { arg lo = 0.00, hi = 1.0, curve = -4;
		^if (this.signalRange == \bipolar) {
			this.lincurve(-1, 1, lo, hi, curve, nil)
		} {
			this.lincurve(0, 1, lo, hi, curve, nil)
		};
	}

	unipolar { arg mul = 1;
		^this.range(0, mul)
	}

	bipolar { arg mul = 1;
		^this.range(mul.neg, mul)
	}

	clip { arg lo = 0.0, hi = 1.0;
		^if(rate == \demand){
			max(lo, min(hi, this))
		}{
			Clip.perform(Clip.methodSelectorForRate(rate), this, lo, hi)
		}
	}

	fold { arg lo = 0.0, hi = 0.0;
		^if(rate == \demand) {
			this.notYetImplemented(thisMethod)
		} {
			Fold.perform(Fold.methodSelectorForRate(rate), this, lo, hi)
		}
	}
	wrap { arg lo = 0.0, hi = 1.0;
		^if(rate == \demand) {
			this.notYetImplemented(thisMethod)
		} {
			Wrap.perform(Wrap.methodSelectorForRate(rate), this, lo, hi)
		}
	}

	degrad {
		// degree * (pi/180)
		^this * 0.01745329251994329547
	}

	raddeg {
		// radian * (180/pi)
		^this * 57.29577951308232286465
	}

	blend { arg that, blendFrac = 0.5;
		var pan;
		^if (rate == \demand || that.rate == \demand) {
			this.notYetImplemented(thisMethod)
		} {
			pan = blendFrac.linlin(0.0, 1.0, -1, 1);
			if (rate == \audio) {
				^XFade2.ar(this, that, pan)
			};

			if (that.rate == \audio) {
				^XFade2.ar(that, this, pan.neg)
			};

			^LinXFade2.perform(LinXFade2.methodSelectorForRate(rate), this, that, pan)
		}
	}

	minNyquist { ^min(this, SampleRate.ir * 0.5) }

	lag { arg t1=0.1, t2;
		^if(t2.isNil) {
			Lag.multiNew(this.rate, this, t1)
		} {
			LagUD.multiNew(this.rate, this, t1, t2)
		}
	}
	lag2 { arg t1=0.1, t2;
		^if(t2.isNil) {
			Lag2.multiNew(this.rate, this, t1)
		} {
			Lag2UD.multiNew(this.rate, this, t1, t2)
		}
	}
	lag3 { arg t1=0.1, t2;
		^if(t2.isNil) {
			Lag3.multiNew(this.rate, this, t1)
		} {
			Lag3UD.multiNew(this.rate, this, t1, t2)
		}
	}

	lagud { arg lagTimeU=0.1, lagTimeD=0.1;
		^LagUD.multiNew(this.rate, this, lagTimeU, lagTimeD)
	}
	lag2ud { arg lagTimeU=0.1, lagTimeD=0.1;
		^Lag2UD.multiNew(this.rate, this, lagTimeU, lagTimeD)
	}
	lag3ud { arg lagTimeU=0.1, lagTimeD=0.1;
		^Lag3UD.multiNew(this.rate, this, lagTimeU, lagTimeD)
	}

	varlag { arg time=0.1, curvature=0, warp=5, start;
		^VarLag.multiNew(this.rate, this, time, curvature, warp, start)
	}

	slew { arg up = 1, down = 1;
		^Slew.multiNew(this.rate, this, up, down)
	}

	prune { arg min, max, type;
		switch(type,
			\minmax, {
				^this.clip(min, max);
			},
			\min, {
				^this.max(min);
			},
			\max, {
				^this.min(max);
			}
		);
		^this
	}

	snap { arg resolution = 1.0, margin = 0.05, strength = 1.0;
		var round = round(this, resolution);
		var diff = round - this;
		^Select.multiNew(this.rate, abs(diff) < margin, this, this + (strength * diff));
	}

	softRound { arg resolution = 1.0, margin = 0.05, strength = 1.0;
		var round = round(this, resolution);
		var diff = round - this;
		^Select.multiNew(this.rate, abs(diff) > margin, this, this + (strength * diff));
	}

	linlin { arg inMin, inMax, outMin, outMax, clip = \minmax;
		if (this.rate == \audio) {
			^LinLin.ar(this.prune(inMin, inMax, clip), inMin, inMax, outMin, outMax)
		} {
			^LinLin.kr(this.prune(inMin, inMax, clip), inMin, inMax, outMin, outMax)
		}
	}

	linexp { arg inMin, inMax, outMin, outMax, clip = \minmax;
		^LinExp.multiNew(this.rate, this.prune(inMin, inMax, clip),
						inMin, inMax, outMin, outMax)
	}
	explin { arg inMin, inMax, outMin, outMax, clip = \minmax;
		^(log(this.prune(inMin, inMax, clip)/inMin))
			/ (log(inMax/inMin)) * (outMax-outMin) + outMin; // no separate ugen yet
	}
	expexp { arg inMin, inMax, outMin, outMax, clip = \minmax;
		^pow(outMax/outMin, log(this.prune(inMin, inMax, clip)/inMin)
			/ log(inMax/inMin)) * outMin;
	}

	lincurve { arg inMin = 0, inMax = 1, outMin = 0, outMax = 1, curve = -4, clip = \minmax;
		var grow, a, b, scaled, curvedResult;
		if (curve.isNumber and: { abs(curve) < 0.125 }) {
			^this.linlin(inMin, inMax, outMin, outMax, clip)
		};
		grow = exp(curve);
		a = outMax - outMin / (1.0 - grow);
		b = outMin + a;
		scaled = (this.prune(inMin, inMax, clip) - inMin) / (inMax - inMin);

		curvedResult = b - (a * pow(grow, scaled));

		if (curve.rate == \scalar) {
			^curvedResult
		} {
			^Select.perform(this.methodSelectorForRate, abs(curve) >= 0.125, [
				this.linlin(inMin, inMax, outMin, outMax, clip),
				curvedResult
			])
		}
	}

	curvelin { arg inMin = 0, inMax = 1, outMin = 0, outMax = 1, curve = -4, clip = \minmax;
		var grow, a, b, scaled, linResult;
		if (curve.isNumber and: { abs(curve) < 0.125 }) {
			^this.linlin(inMin, inMax, outMin, outMax, clip)
		};
		grow = exp(curve);
		a = inMax - inMin / (1.0 - grow);
		b = inMin + a;

		linResult = log( (b - this.prune(inMin, inMax, clip)) / a ) * (outMax - outMin) / curve + outMin;

		if (curve.rate == \scalar) {
			^linResult
		} {
			^Select.perform(this.methodSelectorForRate, abs(curve) >= 0.125, [
				this.linlin(inMin, inMax, outMin, outMax, clip),
				linResult
			])
		}
	}

	bilin { arg inCenter, inMin, inMax, outCenter, outMin, outMax, clip=\minmax;
		^Select.perform(this.methodSelectorForRate, this < inCenter,
			[
				this.linlin(inCenter, inMax, outCenter, outMax, clip),
				this.linlin(inMin, inCenter, outMin, outCenter, clip)
			]
		)
	}

	moddif { |that = 0.0, mod = 1.0|
		^ModDif.multiNew(this.rate, this, that, mod)
	}

	// Note that this differs from |==| for other AbstractFunctions
	// Other AbstractFunctions write '|==|' into the compound function
	// for the sake of their 'storeOn' (compile string) representation.
	// For UGens, scsynth does not support |==| (same handling --> error).
	// So here, we use '==' which scsynth does understand.
	// Also, BinaryOpUGen doesn't write a compile string.
	|==| { |that|
		^this.composeBinaryOp('==', that)
	}
	prReverseLazyEquals { |that|
		// commutative, so it's OK to flip the operands
		^this.composeBinaryOp('==', that)
	}

	sanitize {
		^Sanitize.perform(this.methodSelectorForRate, this);
	}

	signalRange { ^\bipolar }
	@ { arg y; ^Point.new(this, y) } // dynamic geometry support

	addToSynth {
		synthDef = buildSynthDef;
		if (synthDef.notNil, { synthDef.addUGen(this) });
	}

	collectConstants {
		inputs.do({ arg input;
			if (input.isNumber, { synthDef.addConstant(input.asFloat)  });
		});
	}

	isValidUGenInput { ^true }
	asUGenInput { ^this }
	asControlInput { Error("can't set a control to a UGen").throw }
	numChannels { ^1 }


	checkInputs { ^this.checkValidInputs }
	checkValidInputs {
		inputs.do({arg in,i;
			var argName;
			if(in.isValidUGenInput.not,{
				argName = this.argNameForInputAt(i) ? i;
				^"arg: '" ++ argName ++ "' has bad input:" + in;
			})
		});
		^nil
	}

	checkNInputs { arg n;
		if (rate == 'audio') {
			n.do {| i |
				if (inputs.at(i).rate != 'audio') {
					//"failed".postln;
					^("input " ++ i ++ " is not audio rate: " + inputs.at(i) + inputs.at(0).rate);
				};
			};
		};
		^this.checkValidInputs
	}

	checkSameRateAsFirstInput {
		if (rate !== inputs.at(0).rate) {
			^("first input is not" + rate + "rate: " + inputs.at(0) + inputs.at(0).rate);
		};
		^this.checkValidInputs
	}

	argNameForInputAt { arg i;
		var method = this.class.class.findMethod(this.methodSelectorForRate);
		if(method.isNil or: {method.argNames.isNil},{ ^nil });
		^method.argNames.at(i + this.argNamesInputsOffset)
	}
	argNamesInputsOffset { ^1 }
	dumpArgs {
		" ARGS:".postln;
		inputs.do({ arg in,ini;
			("   " ++ (this.argNameForInputAt(ini) ? ini.asString)++":" + in + in.class).postln
		});
	}
	degreeToKey { arg scale, stepsPerOctave=12;
		^DegreeToKey.kr(scale, this, stepsPerOctave)
	}

	outputIndex { ^0 }
	writesToBus { ^false }
	isUGen { ^true }

	poll { arg trig = 10, label, trigid = -1;
		^Poll(trig, this, label, trigid)
	}

	dpoll { arg label, run = 1, trigid = -1;
		^Dpoll(this, label, run, trigid)
	}

	checkBadValues { arg id = 0, post = 2;
		// add the UGen to the tree but keep "this" as the output
		CheckBadValues.perform(this.methodSelectorForRate, this, id, post);
	}

	*methodSelectorForRate { arg rate;
		if(rate == \audio,{ ^\ar });
		if(rate == \control, { ^\kr });
		if(rate == \scalar, {
			if(this.respondsTo(\ir),{
				^\ir
			},{
				^\new
			});
		});
		if(rate == \demand, { ^\new });
		^nil
	}

	*replaceZeroesWithSilence { arg array;
		// this replaces zeroes with audio rate silence.
		// sub collections are deep replaced
		var numZeroes, silentChannels, pos = 0;

		numZeroes = array.count({ arg item; item == 0.0 });
		if (numZeroes == 0, { ^array });

		silentChannels = Silent.ar(numZeroes).asCollection;
		array.do({ arg item, i;
			var res;
			if (item == 0.0, {
				array.put(i, silentChannels.at(pos));
				pos = pos + 1;
			}, {
				if(item.isSequenceableCollection, {
					res = this.replaceZeroesWithSilence(item);
					array.put(i, res);
				});
			});
		});
		^array;
	}


	// PRIVATE
	// function composition
	composeUnaryOp { arg aSelector;
		^UnaryOpUGen.new(aSelector, this)
	}
	composeBinaryOp { arg aSelector, anInput;
		if (anInput.isValidUGenInput, {
			^BinaryOpUGen.new(aSelector, this, anInput)
		},{
			^anInput.performBinaryOpOnUGen(aSelector, this);
		});
	}
	reverseComposeBinaryOp { arg aSelector, aUGen;
		^BinaryOpUGen.new(aSelector, aUGen, this)
	}
	composeNAryOp { arg aSelector, anArgList;
		^thisMethod.notYetImplemented
	}

	// complex support

	asComplex { ^Complex.new(this, 0.0) }
	performBinaryOpOnComplex { arg aSelector, aComplex; ^aComplex.perform(aSelector, this.asComplex) }

	if { arg trueUGen, falseUGen;
		^(this * (trueUGen - falseUGen)) + falseUGen;
	}

	rateNumber {
		if (rate == \audio, { ^2 });
		if (rate == \control, { ^1 });
		if (rate == \demand, { ^3 });
		^0 // scalar
	}
	methodSelectorForRate {
		if(rate == \audio,{ ^\ar });
		if(rate == \control, { ^\kr });
		if(rate == \scalar, {
			if(this.class.respondsTo(\ir),{
				^\ir
			},{
				^\new
			});
		});
		if(rate == \demand, { ^\new });
		^nil
	}
	writeInputSpec { arg file, synthDef;
		file.putInt32(synthIndex);
		file.putInt32(this.outputIndex);
	}
	writeOutputSpec { arg file;
		file.putInt8(this.rateNumber);
	}
	writeOutputSpecs { arg file;
		this.writeOutputSpec(file);
	}
	numInputs { ^inputs.size }
	numOutputs { ^1 }

	name {
		^this.class.name.asString;
	}
	writeDef { arg file;
		try {
			file.putPascalString(this.name);
			file.putInt8(this.rateNumber);
			file.putInt32(this.numInputs);
			file.putInt32(this.numOutputs);
			file.putInt16(this.specialIndex);
			// write wire spec indices.
			inputs.do({ arg input;
				input.writeInputSpec(file, synthDef);
			});
			this.writeOutputSpecs(file);
		} {
			arg e;
			Error("UGen: could not write def: %".format(e.what())).throw;
		}
	}

	initTopoSort {
		inputs.do({ arg input;
			if (input.isKindOf(UGen), {
				antecedents.add(input.source);
				input.source.descendants.add(this);
			});
		});

		widthFirstAntecedents.do({ arg ugen;
			antecedents.add(ugen);
			ugen.descendants.add(this);
		})
	}

	makeAvailable {
		if (antecedents.size == 0, {
			synthDef.available = synthDef.available.add(this);
		});
	}

	removeAntecedent { arg ugen;
		antecedents.remove(ugen);
		this.makeAvailable;
	}

	schedule { arg outStack;
		descendants.reverseDo({ arg ugen;
			ugen.removeAntecedent(this);
		});
		^outStack.add(this);
	}

	optimizeGraph {}

	dumpName {
		^synthIndex.asString ++ "_" ++ this.class.name.asString
	}

	performDeadCodeElimination {
		if (descendants.size == 0) {
			this.inputs.do {|a|
				if (a.isKindOf(UGen)) {
					a.descendants.remove(this);
					a.optimizeGraph
				}
			};
			buildSynthDef.removeUGen(this);
			^true;
		};
		^false
	}
}

// ugen which has no side effect and can therefore be considered for a dead code elimination
// read access to buffers/busses are allowed

PureUGen : UGen {
	optimizeGraph {
		super.performDeadCodeElimination
	}
}

MultiOutUGen : UGen {
	// a class for UGens with multiple outputs
	var <channels;

	*newFromDesc { arg rate, numOutputs, inputs;
		^super.new.rate_(rate).inputs_(inputs).initOutputs(numOutputs, rate)
	}

	initOutputs { arg numChannels, rate;
		if(numChannels.isNil or: { numChannels < 1 }, {
			Error("%: wrong number of channels (%)".format(this, numChannels)).throw
		});
		channels = Array.fill(numChannels, { arg i;
			OutputProxy(rate, this, i);
		});
		if (numChannels == 1, {
			^channels.at(0)
		});
		^channels
	}

	numOutputs { ^channels.size }
	writeOutputSpecs { arg file;
		channels.do({ arg output; output.writeOutputSpec(file); });
	}
	synthIndex_ { arg index;
		synthIndex = index;
		channels.do({ arg output; output.synthIndex_(index); });
	}

}

PureMultiOutUGen : MultiOutUGen {
	optimizeGraph {
		super.performDeadCodeElimination
	}
}

OutputProxy : UGen {
	var <>source, <>outputIndex, <>name;
	*new { arg rate, itsSourceUGen, index;
		^super.new1(rate, itsSourceUGen, index)
	}
	addToSynth {
		synthDef = buildSynthDef;
	}
	init { arg argSource, argIndex;
		source = argSource;
		outputIndex = argIndex;
		synthIndex = source.synthIndex;
	}

	dumpName {
		^this.source.dumpName ++ "[" ++ outputIndex ++ "]"
	}

	controlName {
		var counter = 0, index = 0;

		this.synthDef.children.do({
			arg ugen;
			if(this.source.synthIndex == ugen.synthIndex,
				{ index = counter + this.outputIndex; });
			if(ugen.isKindOf(Control),
				{ counter = counter + ugen.channels.size; });
		});

		^synthDef.controlNames.detect({ |c| c.index == index });
	}

	spec_{ arg spec;
		var controlName, name;
		controlName = this.controlName;
		if (this.controlName.notNil) {
			controlName.spec = spec;
		} {
			"Cannot set spec on a non-Control".error;
		}
	}

}

        
class SawPTR(UGen):
    
    @classmethod
    def ar(cls, freq, phs, sync):
        return cls('audio', 1, [freq, phs, sync])

    @classmethod
    def kr(cls, freq, phs, sync):
        return cls('control', 1, [freq, phs, sync])



        

# the arguments can get numpy arrays, Ugens with kr or ar, or constants
# if a numpy array is given it is taken as audio rate except if it is overriden as kr



for k in range(100):
    freq[:] = np.ones(128, dtype=np.float32) * 400
    unit.calc(64)
    res[k*64:k*64+64] = unit.output_buffers[0][:]



/home/berlach/.local/share/SuperCollider/Extensions/BPartConv.so

/home/berlach/.local/share/SuperCollider/Extensions/ControlMatrix.so
/home/berlach/.local/share/SuperCollider/Extensions/FDN.so


/home/berlach/.local/share/SuperCollider/Extensions/IDNoise.so
/home/berlach/.local/share/SuperCollider/Extensions/Klank2.so
/home/berlach/.local/share/SuperCollider/Extensions/NFilter.so
/home/berlach/.local/share/SuperCollider/Extensions/OnePolePair.so

/home/berlach/.local/share/SuperCollider/Extensions/PAGrains.so
/home/berlach/.local/share/SuperCollider/Extensions/PivawaSync.so

/home/berlach/.local/share/SuperCollider/Extensions/PVOC4.so
/home/berlach/.local/share/SuperCollider/Extensions/PVW.so
/home/berlach/.local/share/SuperCollider/Extensions/ReverbUGens.so
/home/berlach/.local/share/SuperCollider/Extensions/RMSSmooth.so
/home/berlach/.local/share/SuperCollider/Extensions/RSeed.so
/home/berlach/.local/share/SuperCollider/Extensions/SuperMod.so
