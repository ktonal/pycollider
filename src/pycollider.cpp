#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <map>
#include <string>
#include "SC_PlugIn.h"
#include "SC_UnitSpec.h"
#include "SC_WorldOptions.h"
#include "SC_Unit.h"
#include "SC_UnitDef.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

Unit* Unit_New(World* inWorld, UnitSpec* inUnitSpec);
void InterfaceTable_Init();
void Rate_Init(Rate* inRate, double inSampleRate, int inBufLength);
void World_SetSampleRate(World* inWorld, double inSampleRate);

typedef void (*func_ptr_t)(InterfaceTable *);

extern InterfaceTable gInterfaceTable;
extern UnitDef gUnitDefs[16384];
extern int gUnitDefCounter;
std::map<std::string, int> gUGenLookup;
WorldOptions world_options;
World *world;


void
register_buffer(int bufnum, py::array_t<float> array)
{
    SndBuf *buf = world->mSndBufs + bufnum;
    auto buf_info = array.request();
    auto buf_data = static_cast<float*>(buf_info.ptr);

    buf->data = buf_data;
    buf->channels = buf_info.shape[0];
    buf->samples = buf_info.shape[1] * buf_info.shape[0];
    buf->frames = buf_info.shape[1];
    //auto mask = NEXTPOWEROFTWO(buf->samples);  // round up to next power of two
    buf->mask = buf->samples - 1;
    std::cout << "num bufs: " << world->mNumSndBufs << " " << buf->channels << " " << buf->samples << "\n";
}


void pycollider_init()
{
    InterfaceTable_Init();
    world_options = WorldOptions();
    world = World_New(&world_options);
    World_SetSampleRate(world, 44100);
}


void set_buffer_length(int len)
{
    world_options.mBufLength = len;
}


int get_buffer_length()
{
    return world_options.mBufLength;
}


void set_samplerate(int sr)
{
    World_SetSampleRate(world, sr);
}


double get_samplerate()
{
    return world->mSampleRate;
}


void pycollider_load_plugin(char *path)
{
    void *handle;
    func_ptr_t ptr;
    char *error;

    handle = dlopen(path, RTLD_LAZY);
    if (!handle) {
	fputs(dlerror(), stderr);
	exit(1);
    }
    
    ptr = (func_ptr_t) dlsym(handle, "load");
    if ((error = dlerror()) != NULL)  {
	fputs(error, stderr);
	exit(1);
    }

    (*ptr)(&gInterfaceTable);

    for (int k=0; k<gUnitDefCounter; k++) {
	std::string key = std::string((char*)gUnitDefs[k].mUnitDefName);
	gUGenLookup[key] = k;
	fprintf(stdout, "%s\n", (char*) gUnitDefs[k].mUnitDefName);
    }
}


int find_ugen(std::string name)
{
    return gUGenLookup[name];
}


Unit*
make_unit(int ugen_id, py::list inputs, py::list outputs, py::list input_rates, py::list output_rates, int special_index)
{
    // to call the Ctor the unit and all the
    // input/output wires have to be created before
    UnitSpec spec;
    spec.mUnitDef = gUnitDefs + ugen_id;
    spec.mCalcRate = calc_FullRate;
    spec.mNumInputs = inputs.size();
    spec.mNumOutputs = outputs.size();
    spec.mSpecialIndex = special_index;
    spec.mRateInfo = &world->mFullRate;

    // create wires and buffer
    Unit* unit = Unit_New(world, &spec);
    for (unsigned int i=0; i<unit->mNumOutputs; i++) {
	Wire* wire = (Wire*) malloc(sizeof(Wire));
	std::string r = output_rates[i].cast<std::string>();
	if (r == "k") {
	    wire->mCalcRate = calc_BufRate;
	} else {
	    wire->mCalcRate = calc_FullRate;
	}
	unit->mOutput[i] = wire;
	auto out_arr = py::array_t<float>(outputs[i]);
	auto out = static_cast<float*>(out_arr.request().ptr);
	unit->mOutBuf[i] = out;
    }
    for (unsigned int i=0; i<unit->mNumInputs; i++) {
	Wire* wire = (Wire*) malloc(sizeof(Wire));
	std::string r = input_rates[i].cast<std::string>();
	if (r == "k") {
	    wire->mCalcRate = calc_BufRate;
	} else {
	    wire->mCalcRate = calc_FullRate;
	}
	unit->mInput[i] = wire;
	auto in_arr = py::array_t<float>(inputs[i]);
	auto in = static_cast<float*>(in_arr.request().ptr);
	unit->mInBuf[i] = in;
    }
    
    unit->mUnitDef->mUnitCtorFunc(unit);
    return unit;
}


struct PyUnit {
    PyUnit(int ugen_id, py::list inputs, py::list outputs, py::list input_rates, py::list output_rates, int special_index)
    {
	std::cout << "Ctor.\n";
	unit = make_unit(ugen_id, inputs, outputs, input_rates, output_rates, special_index);
    }

    ~PyUnit() { std::cout << "Dtor<" << this << ">\n"; }

    void calc(int inNumSamples)
    {
	unit->mCalcFunc(unit, inNumSamples);
    }
    
    Unit *unit;
};




PYBIND11_MODULE(loader, m) {
    m.doc() = "Load SuperCollider Plugins."; // optional module docstring

    py::class_<PyUnit>(m, "Unit")
	.def(py::init<int, py::list, py::list, py::list, py::list, int>())
	.def("calc", &PyUnit::calc);

    m.def("init", &pycollider_init, "Initialize");
    m.def("load_plugin", &pycollider_load_plugin, "Load a Plugin");
    m.def("register_buffer", &register_buffer, "Make a numpy array available as buffer");
    m.def("find_ugen", &find_ugen, "Find a UGen by name");
    m.def("get_buffer_length", &get_buffer_length, "Returns the current buffer length");
}
