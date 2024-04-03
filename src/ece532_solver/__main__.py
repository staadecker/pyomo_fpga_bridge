import argparse
from pathlib import Path

from ece532_solver.ethernet_interface import (
    transfer_to_simulated_solver,
    transfer_to_fpga,
)
from .mps_reader import MPSReader
from .model_transformations import (
    build_tableau,
    export_to_lp_file,
    transform_into_standard_form,
)


def main():
    # Parse command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path of input file")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Specify an output text file to store the log output.",
        default=None,
    )
    args = parser.parse_args()
    main_without_argument_parser(args.input_file, args.output_file)


def main_without_argument_parser(input_file, output_file=None, use_fpga=False):
    # input_file = "gurobi.mps"
    print("Solving model:", input_file)
    if output_file is None:
        output_file = input_file[:-4] + "_results.txt"

    working_dir = Path(input_file).parent

    # Read input file and load into Model object
    model = MPSReader(input_file).read()

    transform_into_standard_form(model)
    export_to_lp_file(model, working_dir / "standard_form.lp")
    tableau = build_tableau(model)
    print("Generated tableau with shape:", tableau.shape)

    
    if use_fpga:
        transfer_to_fpga(tableau)
    else:
        tableau_file = working_dir / "tableau.bin"
        transfer_to_simulated_solver(tableau, tableau_file)


if __name__ == "__main__":
    main_without_argument_parser("presolved.mps")
