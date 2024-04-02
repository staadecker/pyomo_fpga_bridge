from collections import defaultdict
from typing import List, Type, Optional
from scipy.sparse import coo_array
import numpy as np
import abc

from tqdm import tqdm

from .lp_model import LPModel, Bound, Row


BIG_M = 1e20
TOL = 1e-10


class Transform(abc.ABC):
    def __init__(self, model: LPModel) -> None:
        self.model = model

    def apply(self) -> Optional[str]:
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class MultiTransform(Transform):
    def __init__(self, model: LPModel, *transforms: Type[Transform]) -> None:
        super().__init__(model)
        self.transforms: List[Transform] = [
            transform(model) for transform in transforms
        ]

    def apply(self):
        log = []
        for transform in tqdm(self.transforms, desc="Applying transforms", ascii=True):
            result = transform.apply()
            if result:
                log.append(result)
        print("\n".join(log))

    def undo(self):
        for transform in tqdm(self.transforms, desc="Undoing transforms", ascii=True):
            transform.undo()


class MultiTryTransform(MultiTransform):
    def __init__(self, model: LPModel, *transforms: Type[Transform]) -> None:
        super().__init__(model)
        self.transforms = transforms
        self.applied_transforms = []

    def apply(self):
        while True:
            got_result = False
            for transform in self.transforms:
                transform_instance = transform(self.model)
                result = transform_instance.apply()
                self.applied_transforms.append(transform_instance)
                if result:
                    got_result = True
                    print(result)
            if not got_result:
                break

    def undo(self):
        for transform in self.applied_transforms:
            transform.undo()


class RemoveEmptyConstraints(Transform):
    def apply(self):
        to_delete = []
        for name, row in self.model.constraints:
            if len(row.coefficients) > 0:
                continue
            elif row.row_type == "E":
                assert abs(row.rhs_value) <= TOL, f"RHS value should be zero: {row}"
            elif row.row_type == "L":
                assert row.rhs_value >= -TOL, f"RHS value should be non-negative: {row}"
            elif row.row_type == "G":
                assert row.rhs_value <= TOL, f"RHS value should be non-positive: {row}"
            else:
                raise ValueError(f"Unknown row type: {row.row_type}")
            to_delete.append(name)
        for name in to_delete:
            del self.model.constraints_and_obj[name]
        if to_delete:
            return f"Removed {len(to_delete)} empty constraints"


class RemoveVarsFixedByConstraint(Transform):
    """Removes variables that are fixed due to a constraint like 3 * x = 5"""

    def apply(self):
        rows_to_delete = []
        for _, row in self.model.constraints:
            if row.row_type != "E" or len(row.coefficients) != 1:
                continue
            var_name, coef = list(row.coefficients.items())[0]
            assert coef != 0, f"Coefficient should not be zero: {row}"
            val = row.rhs_value / coef
            self.model.var_results[var_name] = val

            # Replace variable
            for sub_row in self.model.constraints_and_obj.values():
                if var_name in sub_row.coefficients:
                    sub_row.rhs_value -= val * sub_row.coefficients[var_name]
                    del sub_row.coefficients[var_name]
            rows_to_delete.append(row.name)
        for name in rows_to_delete:
            del self.model.constraints_and_obj[name]
        if rows_to_delete:
            return f"Removed {len(rows_to_delete)} fixed variables"


class RemoveVarsFixedByBound(Transform):
    """Removes variables that are fixed due to a bound like 5 <= x <= 5"""

    def apply(self):
        bounds_to_delete = []
        for var_name, bound in self.model.var_bounds.items():
            if (
                bound.lhs_bound is None
                or bound.rhs_bound is None
                or bound.lhs_bound != bound.rhs_bound
            ):
                continue
            val = bound.lhs_bound
            assert var_name == bound.name
            bounds_to_delete.append(var_name)

            self.model.var_results[var_name] = val

            # Replace variable
            for sub_row in self.model.constraints_and_obj.values():
                if var_name in sub_row.coefficients:
                    sub_row.rhs_value -= val * sub_row.coefficients[var_name]
                    del sub_row.coefficients[var_name]
        for name in bounds_to_delete:
            del self.model.var_bounds[name]
        if bounds_to_delete:
            return (
                f"Removed {len(bounds_to_delete)} variables that were fixed by bounds"
            )


class ConstraintToBound(Transform):
    """Converts an inequality constraint with only one variable to a bound."""

    def apply(self):
        rows_to_delete = []
        for name, row in self.model.constraints:
            if len(row.coefficients) != 1 or row.row_type not in ("L", "G"):
                continue

            var_name, coef = list(row.coefficients.items())[0]
            bound_val = row.rhs_value / coef
            if var_name not in self.model.var_bounds:
                self.model.var_bounds[var_name] = Bound(var_name)
            if row.row_type == "L":
                self.model.var_bounds[var_name].tighten(upper_bound=bound_val)
            elif row.row_type == "G":
                self.model.var_bounds[var_name].tighten(lower_bound=bound_val)

            rows_to_delete.append(name)

        for name in rows_to_delete:
            del self.model.constraints_and_obj[name]
        if rows_to_delete:
            return f"Replaced {len(rows_to_delete)} constraints with bounds"

    def undo(self):
        pass


class RemoveUnusedBounds(Transform):
    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.removed_bounds = []

    def apply(self):
        to_delete = []
        for bound in self.model.var_bounds:
            in_use = False
            for row in self.model.constraints_and_obj.values():
                if bound in row.coefficients:
                    in_use = True
                    break
            if not in_use:
                to_delete.append(bound)
        for bound in to_delete:
            self.removed_bounds.append(self.model.var_bounds.pop(bound))
        if to_delete:
            return f"Removed {len(to_delete)} unused bounds"

    def undo(self):
        for bound in self.removed_bounds:
            val = self.model.var_results[bound.name]
            assert bound.lhs_bound is None or bound.lhs_bound <= val
            assert bound.rhs_bound is None or bound.rhs_bound >= val


class RemoveTwoTermEqualities(Transform):
    """Converts ax + by = c to y = c/b - a/b * x and replaces all instances of y with this expression.
    For example if another expression has 5y + 2x < 3, it will be replaced with (2 - 5*(a/b))x < 3 - 5*(c/b)
    """

    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.substitutions = {}

    def apply(self):
        rows_to_delete = []
        for name, row in self.model.constraints:
            if row.row_type != "E" or len(row.coefficients) != 2:
                continue

            (var_x, coef_a), (var_y, coef_b) = row.coefficients.items()

            assert coef_a != 0 and coef_b != 0

            for sub_row in self.model.constraints_and_obj.values():
                if var_y in sub_row.coefficients:
                    # Update coef for x
                    sub_row_coef = sub_row.coefficients.get(var_x, 0)
                    sub_row_coef -= sub_row.coefficients[var_y] * coef_a / coef_b
                    if sub_row_coef == 0:
                        del sub_row.coefficients[var_x]
                    else:
                        sub_row.coefficients[var_x] = sub_row_coef

                    # Update rhs value
                    sub_row.rhs_value -= (
                        sub_row.coefficients[var_y] * row.rhs_value / coef_b
                    )

                    del sub_row.coefficients[var_y]

            assert var_y not in self.substitutions
            self.substitutions[var_y] = (var_y, coef_a, coef_b, row.rhs_value, var_x)
            rows_to_delete.append(name)
        if rows_to_delete:
            return f"Removed {len(rows_to_delete)} two-term equalities"

    def undo(self):
        for var_y, coef_a, coef_b, rhs_value, var_x in self.substitutions.values():
            self.model.var_results[var_y] = (
                rhs_value / coef_b - coef_a / coef_b * self.model.var_results[var_x]
            )


class ReplaceUnboundedVariables(Transform):
    def apply(self):
        """Replaces an unbounded x1 with x1_pos and x1_neg, where x1 = x1_pos - x1_neg and x1_pos, x1_neg >= 0."""
        # Create a mapping from unbounded_vars to their x1_pos and x1_neg counterparts
        self.unbounded_vars = {
            var_name: (f"{var_name}__pos", f"{var_name}__neg")
            for row in self.model.constraints_and_obj.values()
            for var_name in row.coefficients.keys()
            if var_name not in self.model.var_bounds
        }

        # Add bounds
        for pos_var, neg_var in self.unbounded_vars.values():
            self.model.var_bounds[pos_var] = Bound(pos_var, lhs_bound=0)
            self.model.var_bounds[neg_var] = Bound(neg_var, lhs_bound=0)

        # Replace unbounded variables with x1_pos and x1_neg in the constraints
        for var_name, (pos_var, neg_var) in self.unbounded_vars.items():
            for row in self.model.constraints_and_obj.values():
                if var_name in row.coefficients:
                    row.coefficients[pos_var] = row.coefficients[var_name]
                    row.coefficients[neg_var] = -row.coefficients[var_name]
                    del row.coefficients[var_name]

    def undo(self):
        for var_name, (pos_var, neg_var) in self.unbounded_vars.items():
            self.model.var_results[var_name] = (
                self.model.var_results[pos_var] - self.model.var_results[neg_var]
            )
            del self.model.var_results[pos_var]
            del self.model.var_results[neg_var]


class FlipUpperBounds(Transform):
    """Converts x <= 5 to x_inv >= -5 where x_inv = -x."""

    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.flipped_vars = {}

    def apply(self):
        new_bounds = {}
        for var, bound in self.model.var_bounds.items():
            if not (bound.rhs_bound is not None and bound.lhs_bound is None):
                continue
            flipped_var = f"{var}__inv"
            # Multiply variable by negative 1
            for row in self.model.constraints_and_obj.values():
                if var in row.coefficients:
                    row.coefficients[flipped_var] = -row.coefficients.pop(var)
            new_bounds[flipped_var] = Bound(flipped_var, lhs_bound=-bound.rhs_bound)
            self.flipped_vars[var] = flipped_var
        for var, flipped_var in self.flipped_vars.items():
            del self.model.var_bounds[var]
            self.model.var_bounds[flipped_var] = new_bounds[flipped_var]

    def undo(self):
        for var, flipped_var in self.flipped_vars.items():
            self.model.var_results[var] = -self.model.var_results[flipped_var]
            del self.model.var_results[flipped_var]


class MoveUpperBoundToConstraint(Transform):
    def apply(self):
        """Converts a<=x<=b to x>=a with constraint x<=b."""
        for var, bound in self.model.var_bounds.items():
            if bound.rhs_bound is None or bound.lhs_bound is None:
                continue

            const_name = f"{var}__upper_bound"
            assert const_name not in self.model.constraints_and_obj
            self.model.constraints_and_obj[const_name] = Row(
                row_name=const_name,
                row_type="L",
                coefficients={var: 1.0},
                rhs_value=bound.rhs_bound,
            )
            bound.rhs_bound = None

    def undo(self): ...


class ShiftBoundsToZero(Transform):
    """Converts 4<=x to 0<=x' where x' = x-4 (or x = x' + 4). In the matrix this means adding 4 * coef to the RHS value. TODO check"""

    def apply(self):
        self.shifted_vars = {
            var: (f"{var}__shift", bound.lhs_bound)
            for var, bound in self.model.var_bounds.items()
            if bound.lhs_bound is not None and bound.lhs_bound != 0
        }

        for var, (shift_var, shift_amount) in self.shifted_vars.items():
            for row in self.model.constraints_and_obj.values():
                if var in row.coefficients:
                    row.rhs_value += shift_amount * row.coefficients[var]
                    row.coefficients[shift_var] = row.coefficients[var]
                    del row.coefficients[var]

            self.model.var_bounds[shift_var] = Bound(shift_var, lhs_bound=0)
            del self.model.var_bounds[var]

    def undo(self):
        for var, (shift_var, shift_amount) in self.shifted_vars.items():
            self.model.var_results[var] = (
                self.model.var_results[shift_var] + shift_amount
            )
            del self.model.var_results[shift_var]


class AddArtificialVariables(Transform):
    """Transforms constraints into equalities with an artificial variable and slack variables"""

    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.artificial_vars = []

    def apply(self):
        for row in self.model.constraints_and_obj.values():
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
                self.artificial_vars.append(artificial_name)
                row.coefficients[artificial_name] = 1
                self.model.objective.coefficients[artificial_name] = BIG_M
                self.model.var_bounds[artificial_name] = Bound(
                    artificial_name, lhs_bound=0
                )

            # If we have <= or >= we need to add a slack variable
            if row.row_type in ("L", "G"):
                slack_name = f"slack_{row.name}"
                self.artificial_vars.append(slack_name)
                row.coefficients[slack_name] = 1 if row.row_type == "L" else -1
                self.model.var_bounds[slack_name] = Bound(slack_name, lhs_bound=0)
                row.row_type = "E"

    def undo(self):
        for var in self.artificial_vars:
            # TODO use these values to find the constraints' duals?
            del self.model.var_results[var]


def check_in_standard_form(model: LPModel):
    vars = set()
    for row in model.constraints_and_obj.values():
        assert row.rhs_value >= 0, "RHS value must be non-negative"
        assert (
            row.row_type == "E" or row.row_type == "N"
        ), "All constraints must be equality constraints or the objective"
        for var in row.coefficients.keys():
            vars.add(var)

    for var in vars:
        assert var in model.var_bounds, "All variables must have bounds"
        bound = model.var_bounds[var]
        assert bound.lhs_bound == 0, "All variables must have zero lower bounds"
        assert bound.rhs_bound is None, "All variables must have no upper bounds"


def transform_into_standard_form(model):
    export_to_lp_file(model, "before_presolve.lp")
    presolve_transforms = MultiTryTransform(
        model,
        RemoveEmptyConstraints,
        ConstraintToBound,
        RemoveVarsFixedByConstraint,
        RemoveVarsFixedByBound,
        RemoveTwoTermEqualities,
        RemoveUnusedBounds,
    )
    presolve_transforms.apply()
    export_to_lp_file(model, "after_presolve.lp")
    transforms = MultiTransform(
        model,
        MoveUpperBoundToConstraint,
        FlipUpperBounds,
        ReplaceUnboundedVariables,
        ShiftBoundsToZero,
        AddArtificialVariables,
    )
    transforms.apply()
    export_to_lp_file(model, "after_standard_form.lp")
    check_in_standard_form(model)


def export_to_lp_file(model: LPModel, filename: str):
    with open(filename, "w") as f:
        f.write("Minimize\n")
        f.write("obj: ")
        for var, coef in model.objective.coefficients.items():
            f.write(f"{coef:+.12g} {var} ")

        f.write("\nSubject To\n")
        for _, row in model.constraints:
            f.write(f"{row}\n")
        f.write("\nBounds\n")
        for var, bound in model.var_bounds.items():
            # Doing + 0. to avoid -0.0, in Python 3.11 and up the format specifier z can be used (see PEP682)
            if bound.lhs_bound is None:
                lhs = "-Inf <= "
            elif bound.lhs_bound == 0:
                lhs = ""
            else:
                lhs = f"{bound.lhs_bound + 0:.12g} <= "
            if bound.rhs_bound is None:
                rhs = ""
            else:
                rhs = f" <= {bound.rhs_bound + 0:.12g}"
            if lhs or rhs:
                f.write(f"{lhs}{var}{rhs}\n")
        free_variables = []
        for row in model.constraints_and_obj.values():
            for var in row.coefficients:
                if var not in free_variables and var not in model.var_bounds:
                    free_variables.append(var)
        for free_variable in free_variables:
            f.write(f"{free_variable} free\n")
        f.write("\nEnd\n")


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
        shape=(len(model.constraints_and_obj), len(model.var_bounds)),
        dtype=np.float64,
    ).toarray()
