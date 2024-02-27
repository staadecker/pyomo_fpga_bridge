import argparse
from .mps_reader import MPSReader
from .model_transformations import build_tableau, transform_into_standard_form


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


def main_without_argument_parser(input_file, output_file=None):
    if output_file is None:
        output_file = input_file[:-4] + "_results.txt"

    # Read input file and load into Model object
    model = MPSReader(input_file).read()
    transform_into_standard_form(model)
    tableau = build_tableau(model)
    print("Generated tableau with shape:", tableau.shape)
    # print(tableau.tobytes())
    
    # Analyze the model

def solve(input_file):
    print("Solving model:", input_file)
    main_without_argument_parser(input_file)

if __name__ == "__main__":
    main_without_argument_parser("presolved.mps")