

UnaryOpUGen

NAryOpFunction

SinOsc.ar(10).clip(0, 1.0)

Clip



~res = Dictionary.new;
[ 'neg', 'reciprocal', 'bitNot', 'abs', 'asFloat', 'asInteger', 'ceil', 'floor',
    'frac', 'sign', 'squared', 'sqrt', 'exp', 'midicps', 'cpsmidi', 'midiratio', 'ratiomidi',
    'ampdb', 'dbamp', 'octcps', 'cpsoct', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin',
    'acos', 'atan', 'sinh', 'cosh', 'tanh', 'rand', 'rand2', 'linrand', 'bilinrand',
    'sum3rand', 'distort', 'softclip', 'coin', 'even', 'odd', 'rectWindow', 'hanWindow',
    'welWindow', 'triWindow', 'scurve', 'ramp', 'isPositive', 'isNegative',
    'isStrictlyPositive', 'isPositive', 'isNegative', 'isStrictlyPositive', 'rho', 'theta',
    'degrad', 'raddeg', '+', '-', '*', '/', 'mod', 'pow', '<', '<=', '>', '>=', 'round',
    'roundUp', 'trunc', 'atan2', 'hypot', 'difsqr', 'sumsqr', 'sqrsum', 'sqrdif', 'absdif',
    'thres', 'amclip', 'scaleneg', 'clip2', 'fold2', 'wrap2', 'rrand', 'exprand' ].do{ arg sym;
		~res[sym] = sym.specialIndex
	}

~res.keysValuesDo{ arg k, v; (k.asCompileString ++ ": " ++ v.asCompileString ++ ",").postln }

'round'.specialIndex
'midiratio'.specialIndex

not done:
'odd': -1,
'isStrictlyPositive': -1,
'isNegative': -1,
'isPositive': -1,
'even': -1,
'theta': -1,
'rho': -1,


'odd'.specialIndex
UGen
SinOsc.ar(10).rectWindow

	degrad {
		// degree * (pi/180)
		^this * 0.01745329251994329547
	}

	raddeg {
		// radian * (180/pi)
		^this * 57.29577951308232286465
	}
