from typing import Dict, Optional, Tuple

# Mapping of row types to user friendly outputs. Used when printing rows.
# RHS values are on the left, hence why >= and <= are flipped.
SYMBOL_MAPPING = {"L": ">=", "G": "<=", "E": "=", "N": "Obj:"}


class LPModel:
    """Represents a linear model. Contains all the rows, variable bounds and objective function."""

    def __init__(self):
        self._obj_name: Optional[str] = None  # Reference to the objective function
        self.constraints_and_obj: Dict[str, Row] = {}  # Will include the objective function
        self.variables: Dict[str, Bound] = {}
        self.minimization_problem: Optional[bool] = None

    def add_row(self, row_name: str, row_type: str):
        assert (
            row_name not in self.constraints_and_obj
        )  # Make sure it doesn't already exist (don't want to overwrite)
        self.constraints_and_obj[row_name] = Row(row_name, row_type)
        if row_type == "N":  # If row is the objective row
            if self._obj_name is not None:
                raise Exception("Can't set objective, it already exists")
            self._obj_name = row_name
            self.constraints_and_obj[row_name].is_objective = True

    @property
    def objective(self):
        return self.constraints_and_obj.get(self._obj_name, None)

    def print_model(self):
        print("OBJECTIVE")
        self.objective.print()
        print("\nCONSTRAINTS")
        for row in self.constraints_and_obj.values():
            if row.row_type != "N":
                row.print()
        print("\nBOUNDS")
        for bound in self.variables.values():
            bound.print()


class Row:
    """A constraint or the objective function in the model."""

    def __init__(self, row_name: str, row_type=None, coefficients=None, rhs_value=0.0):
        """
        :param row_name: name of the row
        :param row_type: letter representing the row type according to the MPS format (must be a key in SYMBOL_MAPPING)
        """
        self.name: str = row_name
        self.row_type: str = row_type
        if coefficients is None:
            coefficients = {}
        self.coefficients: Dict[str, float] = coefficients
        self.rhs_value: Optional[
            float
        ] = rhs_value  # Need float since that's what's expected in analysis
        self.is_objective: bool = False  # Can be overidden after creation

    def print(self):
        print(self.name, end=":\t")
        if self.rhs_value is not None:
            print(self.rhs_value, end="\t")
        print(f"{SYMBOL_MAPPING[self.row_type]} ", end="")
        for var_name, coefficient in self.coefficients.items():
            if coefficient > 0:
                print("+", end="")
            if coefficient == 1:
                print(f"{var_name}", end="\t")
            else:
                print(f"{coefficient}*{var_name}", end="\t")
        print()

    def coefficient_range(
        self,
    ) -> Tuple[Tuple[Optional[str], float], Tuple[Optional[str], float]]:
        """
        Returns two tuples containing the name and value of the minimum and maximum coefficient for this row.
        """
        absolute_coefficients = tuple(
            filter(
                lambda k_v: k_v[1] != 0,
                map(lambda k_v: (k_v[0], abs(k_v[1])), self.coefficients.items()),
            )
        )
        if absolute_coefficients:
            return min(absolute_coefficients, key=lambda k_v: k_v[1]), max(
                absolute_coefficients, key=lambda k_v: k_v[1]
            )
        return (None, float("inf")), (None, 0)


class Bound:
    """A bound on a variable"""

    def __init__(self, name: str, lhs_bound=None, rhs_bound=None):
        self.name: str = name
        self.lhs_bound: Optional[float] = lhs_bound
        self.rhs_bound: Optional[float] = rhs_bound

    def print(self):
        if self.lhs_bound is not None and self.rhs_bound is not None:
            print(self.lhs_bound, "<=", self.name, "<=", self.rhs_bound)
        elif self.rhs_bound is not None:
            print(self.name, "<=", self.rhs_bound)
        elif self.lhs_bound is not None:
            print(self.lhs_bound, "<=", self.name)
        else:
            print("unbounded ", self.name)
