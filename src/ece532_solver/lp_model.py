from typing import Dict, Generator, Optional, Tuple

# Mapping of row types to user friendly outputs. Used when printing rows.
SYMBOL_MAPPING = {"L": "<=", "G": ">=", "E": "=", "N": "Obj:"}


class LPModel:
    """Represents a linear model. Contains all the rows, variable bounds and objective function."""

    def __init__(self, minimization_problem):
        self._obj_name: Optional[str] = None  # Reference to the objective function
        self.constraints_and_obj: Dict[str, Row] = (
            {}
        )  # Will include the objective function
        self.variables: Dict[str, Variable] = {}
        self.minimization_problem: bool = minimization_problem
        self.var_results: Dict[str, float] = {}

    def assert_is_valid(self):
        """Checks that the variables in the model are the same as the variables in the constraints and objective."""
        assert self.minimization_problem is not None, "No sense set"
        variables_in_constr = {
            var for row in self.constraints_and_obj.values() for var in row.coefficients
        }
        variables_in_model = set(self.variables.keys())
        assert variables_in_model == variables_in_constr, f"{len(variables_in_model)} variables in model but {len(variables_in_constr)} in constraints and objective ({variables_in_constr ^ variables_in_model})"

    def assert_is_in_standard_form(self):
        assert not self.minimization_problem
        for _, row in self.constraints:
            assert row.rhs_value >= 0, f"RHS value must be non-negative, {row}"
            assert row.row_type == "E", "All constraints must be equalities"

        self.assert_is_valid()
        for var_name, var in self.variables.items():
            assert var.lhs_bound == 0, f"All variables must have zero lower bounds, {var_name, var.lhs_bound, var.rhs_bound}"
            assert var.rhs_bound is None, f"All variables must have no upper bounds, {var_name, var.lhs_bound, var.rhs_bound}"

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
    def objective(self) -> Optional["Row"]:
        return self.constraints_and_obj.get(self._obj_name, None)

    @property
    def constraints(self) -> Generator[Tuple[str, "Row"], None, None]:
        for name, row in self.constraints_and_obj.items():
            if not row.is_objective:
                yield name, row


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
        self.rhs_value: Optional[float] = (
            rhs_value  # Need float since that's what's expected in analysis
        )
        self.is_objective: bool = False  # Can be overidden after creation

    def __str__(self) -> str:
        s = f"{self.name}: "
        for var_name, coefficient in self.coefficients.items():
            s += f"{coefficient:+.12g} {var_name} "
        s += f"{SYMBOL_MAPPING[self.row_type]} {self.rhs_value}"
        return s

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


class Variable:
    """A variable with its bound"""

    def __init__(self, var_name: str, lhs_bound=None, rhs_bound=None):
        self.name: str = var_name
        self.lhs_bound: Optional[float] = lhs_bound
        self.rhs_bound: Optional[float] = rhs_bound

    def tighten(self, lower_bound=None, upper_bound=None):
        if lower_bound is not None:
            if self.lhs_bound is None:
                self.lhs_bound = lower_bound
            else:
                self.lhs_bound = max(self.lhs_bound, lower_bound)
        if upper_bound is not None:
            if self.rhs_bound is None:
                self.rhs_bound = upper_bound
            else:
                self.rhs_bound = min(self.rhs_bound, upper_bound)

    def print(self):
        if self.lhs_bound is not None and self.rhs_bound is not None:
            print(self.lhs_bound, "<=", self.name, "<=", self.rhs_bound)
        elif self.rhs_bound is not None:
            print(self.name, "<=", self.rhs_bound)
        elif self.lhs_bound is not None:
            print(self.lhs_bound, "<=", self.name)
        else:
            print("unbounded ", self.name)
