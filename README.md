
PyCollider
==========

Load SuperCollider plugins into python.

The package is motivated by two distinct goals:

PyCollider makes it possible to run SuperCollider plugins with numpy
arrays as inputs and outputs.  This simplifies the development of new
synthesis algorithms as numpy/scipy provide a well established
framework for prototyping DSP techniques and tools to plot and analyze
results.  As such the package can be used complementary to
SuperCollider as a development and debugging tool for new plugins.
Because PyCollider loads the plugin binaries, you can be sure that
you are executing the exact same code as when you run the plugins in
SuperCollider.

Teaching the fundamentals of computer-music to composers can be
complicated by the amount of different specialized (and often ancient)
languages and software systems.  To seriously learn about
computer-music, some programming knowledge is necessary.  While this
comes naturally to some students it is an obstacle to others.  By
creating an infrastructure for sound synthesis and basic algorithmic
composition techniques (e.g. the python pbind package) students can
focus on concepts instead of syntactic and semantic idiosyncrasies of
multiple different programming languages.  For DSP and machine
learning python has become one of the most wide-spread languages.
Python, which was not developed to be a responsive low-latency
real-time language will not replace systems like SuperCollider for
real-time electronic performances, but for pedagogical purposes and
offline composition work, a general purpose language, designed to have
a shallow learning curve has some clear advantages.

Installation
============

The package can be installed with

```shell
python setup.py install
```

You need to have SuperCollider plugins installed and load them.
To load a plugin (the binary) use the load_plugin function:

pycollider.load_plugin("/path/to/plugins/SomePlugin.so")


Examples
========

You must make sure to load the plugin files before you create any UGen object.

The following code generates 44100 samples of a 100 Hz sine-wave
(returns a numpy array).

sig = SinOsc(100).render(44100)

Numpy arrays can also be used a arguments.  In this example the previously
generated sine-wave is used as frequency argument to another Oscillator
resulting in a frequency modulated sound.

sig2 = SinOsc(sig).render(44100)

Additional to numpy arrays, unit generators can be used as inputs to other
generators as well to create more complex modular synthesis networks.  The
following code generates a 220 hertz sine-wave multiplied by a 10 Hz Sawtooth
oscillator (some form of amplitude modulation).

sig3 = SinOsc(220, mul=Saw(10).range(0, 1.0)).render(44100)

The unit generators can also be combined by arithmetic operators.  To generate
the sum of two oscillators you can just use '+' to add the outputs together.

sig4 = (SinOsc(100) + SinOsc(110)).render(44100)






