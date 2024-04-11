# Pyomo to FPGA with custom presolve

This tool was part of a [larger project](https://github.com/abnashkb/ece532) for a university course. In this repo lies a solver plugin for [Pyomo](http://www.pyomo.org/) that converts a Pyomo linear programing model into a bytestream that can be sent over ethernet to an FPGA. The bytestream is the serialized representation of a linear programming tableau, in standard form. The tableau is generated using a custom presolve algorithm.

The tool was tested using [this](https://github.com/REAM-lab/Staadecker_et_al_2022_archive) [SWITCH model](https://github.com/REAM-lab/switch). Tests show that the generated tableau, when solved with Gurobi, produce the same results as running SWITCH with Gurobi directly. This confirms, that our tableau generation process works as intended.

## Performant presolve

The presolve includes multiple transforms that have successfully reduced large models by ~10x in size. Compared to Gurobi, the presolve efficiency was similar (better in some cases!).

## License

The code in this repository builds on my code from the [lp-analyzer project](https://github.com/staadecker/lp-analyzer/)
and the Pyomo library. Copyright belongs to those respective libraries when applicable while new code is under the MIT license.