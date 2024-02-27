from collections import defaultdict
from .core import LPModel, Bound, Row
from scipy.sparse import coo_array
import numpy as np

BIG_M = 1e20


def replace_unbounded_variables(model: LPModel):
    """Replaces an unbounded x1 with x1_pos and x1_neg, where x1 = x1_pos - x1_neg and x1_pos, x1_neg >= 0."""
    unbounded_vars = set()
    for row in model.constraints_and_obj.values():
        for var_name in row.coefficients.keys():
            if var_name not in model.variables:
                unbounded_vars.add(var_name)

    for var_name in unbounded_vars:
        pos_var = f"{var_name}_pos"
        neg_var = f"{var_name}_neg"
        for row in model.constraints_and_obj.values():
            if var_name in row.coefficients:
                row.coefficients[pos_var] = row.coefficients[var_name]
                row.coefficients[neg_var] = -row.coefficients[var_name]
                del row.coefficients[var_name]
        model.variables[pos_var] = Bound(pos_var, lhs_bound=0)
        model.variables[neg_var] = Bound(neg_var, lhs_bound=0)


def remove_upper_bounds(model: LPModel):
    """Converts x<=5 to x' >= -5 where x' = -x. Also converts 0<=x<=5 to x>=0 with constraint x<=5."""
    for var, bound in model.variables.items():
        if bound.rhs_bound is None:
            continue
        if bound.lhs_bound is None:
            # Multiply variable by negative 1
            for row in model.constraints_and_obj.values():
                if var in row.coefficients:
                    row.coefficients[var] = -row.coefficients[var]
            bound.lhs_bound = -bound.rhs_bound
            bound.rhs_bound = None
        else:
            bound_name = f"{var}_upper_bound"
            model.constraints_and_obj[bound_name] = Row(
                row_name=bound_name,
                row_type="L",
                coefficients={var: 1},
                rhs_value=bound.rhs_bound,
            )

            bound.rhs_bound = None


def shift_bounds_to_zero(model: LPModel):
    """Converts 4<=x to 0<=x' where x' = x-4. In the matrix this means adding 4 * coef to the RHS value."""
    vars_to_shift = set()
    for var, bound in model.variables.items():
        if bound.lhs_bound is None or bound.lhs_bound == 0:
            continue
        vars_to_shift.add(var)

    for var in vars_to_shift:
        shift_var = f"{var}_shift"

        for row in model.constraints_and_obj.values():
            if var in row.coefficients:
                coef = row.coefficients[var]
                row.rhs_value += bound.lhs_bound * coef
                row.coefficients[shift_var] = coef
                del row.coefficients[var]

        model.variables[shift_var] = Bound(bound.name, lhs_bound=0)
        del model.variables[var]


def add_artificial_variables(model: LPModel):
    for row in model.constraints_and_obj.values():
        if row.is_objective:
            continue
        # Ensure a positive RHS by multiplying by -1
        if row.rhs_value < 0:
            row.coefficients = {
                var_name: -coefficient
                for var_name, coefficient in row.coefficients.items()
            }
            row.rhs_value = -row.rhs_value
            if row.row_type == "L":
                row.row_type = "G"
            elif row.row_type == "G":
                row.row_type = "L"

        # If we have a = or >= we need to add an artificial variable
        if row.row_type in ("E", "G"):
            artificial_name = f"art_{row.name}"
            row.coefficients[artificial_name] = 1
            model.objective.coefficients[artificial_name] = BIG_M
            model.variables[artificial_name] = Bound(artificial_name, lhs_bound=0)

        # If we have <= or >= we need to add a slack variable
        if row.row_type in ("L", "G"):
            slack_name = f"slack_{row.name}"
            row.coefficients[slack_name] = 1 if row.row_type == "L" else -1
            model.variables[slack_name] = Bound(slack_name, lhs_bound=0)
            row.row_type = "E"


def check_in_standard_form(model: LPModel):
    vars = set()
    for row in model.constraints_and_obj.values():
        assert row.rhs_value >= 0, "RHS value must be non-negative"
        assert (
            row.row_type == "E" or row.row_type == "N"
        ), "All constraints must be equality constraints"
        for var in row.coefficients.keys():
            vars.add(var)

    for var in vars:
        assert var in model.variables, "All variables must have bounds"
        bound = model.variables[var]
        assert bound.lhs_bound == 0, "All variables must have zero lower bounds"
        assert bound.rhs_bound is None, "All variables must have no upper bounds"


def transform_into_standard_form(model):
    remove_upper_bounds(model)
    replace_unbounded_variables(model)
    shift_bounds_to_zero(model)
    add_artificial_variables(model)
    check_in_standard_form(model)


def build_tableau(model: LPModel):
    variable_order = defaultdict(lambda: len(variable_order))
    column_order = defaultdict(lambda: len(column_order))

    i = []
    j = []
    data = []

    for row in model.constraints_and_obj.values():
        row_index = column_order[row.name]
        for var, coef in row.coefficients.items():
            col_index = variable_order[var]
            i.append(row_index)
            j.append(col_index)
            data.append(coef)

    return coo_array(
        (data, (i, j)),
        shape=(len(model.constraints_and_obj), len(model.variables)),
        dtype=np.float64,
    ).toarray()
