from collections import defaultdict
from typing import List, Type, Optional
from scipy.sparse import coo_array
import numpy as np
import gurobipy as gp
import abc

from tqdm import tqdm

from .lp_model import LPModel, Variable, Row


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
        for i in range(100):
            results = []
            for transform in tqdm(
                self.transforms, desc=f"Presolve round {i+1}", ascii=True
            ):
                transform_instance = transform(self.model)
                result = transform_instance.apply()
                self.applied_transforms.append(transform_instance)
                if result:
                    results.append(result)
            if not results:
                break
            print("\n".join(results))

    def undo(self):
        for transform in self.applied_transforms:
            transform.undo()


class RemoveEmptyConstraints(Transform):
    def apply(self):
        to_delete = []
        for name, row in self.model.constraints:
            if len(row.coefficients) == 0:
                if row.row_type == "E":
                    assert abs(row.rhs_value) <= TOL, f"RHS value should be zero: {row}"
                elif row.row_type == "L":
                    assert (
                        row.rhs_value >= -TOL
                    ), f"RHS value should be non-negative: {row}"
                elif row.row_type == "G":
                    assert (
                        row.rhs_value <= TOL
                    ), f"RHS value should be non-positive: {row}"
                else:
                    raise ValueError(f"Unknown row type: {row.row_type}")
                to_delete.append(name)
        for name in to_delete:
            del self.model.constraints_and_obj[name]
        if to_delete:
            return f"Removed {len(to_delete)} constraints (empty row)"


class RemoveVarsFixedByConstraint(Transform):
    """Removes variables that are fixed due to a constraint like 3 * x = 5"""

    def apply(self):
        rows_to_delete = []
        for _, row in self.model.constraints:
            if row.row_type == "E" and len(row.coefficients) == 1:
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
            return f"Removed {len(rows_to_delete)} constraints/variables (singleton equality)"


class RemoveVarsFixedByBound(Transform):
    """Removes variables that are fixed due to a bound like 5 <= x <= 5"""

    def apply(self):
        bounds_to_delete = []
        for var_name, bound in self.model.variables.items():
            if not (
                bound.lhs_bound is not None
                and bound.rhs_bound is not None
                and abs(bound.lhs_bound - bound.rhs_bound) <= TOL
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
            del self.model.variables[name]
        if bounds_to_delete:
            return f"Removed {len(bounds_to_delete)} variables (squeezed by bounds)"


class ConstraintToBound(Transform):
    """Converts an inequality constraint with only one variable to a bound."""

    def apply(self):
        rows_to_delete = []
        for name, row in self.model.constraints:
            if not (len(row.coefficients) == 1 and row.row_type in ("L", "G")):
                continue

            var_name, coef = row.coefficients.popitem()
            if coef < 0:
                coef *= -1
                row.rhs_value *= -1
                if row.row_type == "L":
                    row.row_type = "G"
                elif row.row_type == "G":
                    row.row_type = "L"
            bound_val = row.rhs_value / coef
            if var_name not in self.model.variables:
                self.model.variables[var_name] = Variable(var_name)
            if row.row_type == "L":
                self.model.variables[var_name].tighten(upper_bound=bound_val)
            elif row.row_type == "G":
                self.model.variables[var_name].tighten(lower_bound=bound_val)

            rows_to_delete.append(name)

        for name in rows_to_delete:
            del self.model.constraints_and_obj[name]
        if rows_to_delete:
            return f"Removed {len(rows_to_delete)} constraints (singleton inequality)"

    def undo(self):
        pass


class RemoveUnusedVariables(Transform):
    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.removed_bounds = []

    def apply(self):
        to_delete = []
        for bound in self.model.variables:
            in_use = False
            for row in self.model.constraints_and_obj.values():
                if bound in row.coefficients:
                    in_use = True
                    break
            if not in_use:
                to_delete.append(bound)
        for bound in to_delete:
            self.removed_bounds.append(self.model.variables.pop(bound))
        if to_delete:
            return f"Removed {len(to_delete)} variables (unused)"

    def undo(self):
        for bound in self.removed_bounds:
            val = self.model.var_results[bound.name]
            assert bound.lhs_bound is None or bound.lhs_bound <= val
            assert bound.rhs_bound is None or bound.rhs_bound >= val


class PushVarsToBounds(Transform):
    """Identifies variables that only appear in the objective and pushes them to their bounds."""

    def apply(self):
        vars_pushed_to_limit = 0
        for var_name, var in self.model.variables.items():
            if var_name not in self.model.objective.coefficients:
                continue
            if any(
                var_name in row.coefficients
                for _, row in self.model.constraints_and_obj.items()
            ):
                continue

            should_maximize = (self.model.objective.coefficients[var_name] > 0) ^ (
                self.model.minimization_problem
            )

            if should_maximize:
                assert var.rhs_bound is not None, "Problem is unbounded"
                var.lhs_bound = var.rhs_bound
            else:
                assert var.lhs_bound is not None, "Problem is unbounded"
                var.rhs_bound = var.lhs_bound
            vars_pushed_to_limit += 1

        if vars_pushed_to_limit:
            return f"Pushed {vars_pushed_to_limit} variables to their limits"

    def undo(self): ...


class SnapConstraintsToEquality(Transform):
    """Takes a constraint like x + y <= 5 and snaps it to x + y = 5 if the objective is pushing towards that bound.
    Special conditions exist namely:
    - The variable being pushed must not be in any other constraint
    - The constraint must be either <= or >= (obviously)
    - The direction of push should match the direction of the constraint
    - If the variable being pushed also has a bound in the same direction, that bound must be redundant.

    To know if that bound is redundant, we calculate the weakest implied bound by setting all the other variables to their
    weakest bound. For example,
    Say:
    2x + 3y - 4z <= 6
    And y <= 4, z <= 1, x>=0

    We can rearrange the constraint to:
    y <= (6 + 4z - 2x) / 3

    The bound is weakest when z is largest and x is smallest. Thankfully, z and x have largest and smallest bounds.
    At those bounds
    y <= (6 + 4*1 - 2*0) / 3 = 3.33

    Since 3.33 is less than 4, the bound is redundant and y will always be maximized such that the constraint is an equality.
    """
    def apply(self):
        constraints_snapped = 0
        for var_name, var in self.model.variables.items():
            obj_coef = self.model.objective.coefficients.get(var_name, None)
            if obj_coef is None:
                continue

            # If the obj_coef < 0, maximizing the variable minimizes the objective, which is desired for minimization problems.
            should_maximize = (obj_coef < 0) == self.model.minimization_problem

            # Find the constraint row
            row = [
                row for _, row in self.model.constraints if var_name in row.coefficients
            ]
            if len(row) != 1:
                continue
            row = row[0]

            # We are trying to push towards an equality so this is required
            if row.row_type not in ("L", "G"):
                continue

            main_coef = row.coefficients[var_name]

            # An it's an upperbound (e.g. x + y <= 5) but we want to minimize the constraint is irrelevant
            is_upper_bound = (row.row_type == "L") == (main_coef > 0)
            if should_maximize != is_upper_bound:
                continue

            # If our variable has a bound, we can only snap if the bound is redundant.
            var_bound = var.rhs_bound if should_maximize else var.lhs_bound
            if var_bound is not None:
                implied_bound = row.rhs_value
                for term_name, term_coef in row.coefficients.items():
                    if var_name != term_name:
                        term = self.model.variables[term_name]
                        weakest_bound = (
                            term.lhs_bound
                            if is_upper_bound ^ (term_coef < 0) ^ (main_coef < 0)
                            else term.rhs_bound
                        )
                        if weakest_bound is None:
                            implied_bound = None
                            break
                        implied_bound -= term_coef * weakest_bound
                if implied_bound is None:
                    continue
                implied_bound /= main_coef
                if is_upper_bound:
                    is_redundant = implied_bound <= var_bound
                else:
                    is_redundant = implied_bound >= var_bound
                if not is_redundant:
                    continue

            row.row_type = "E"
            constraints_snapped += 1
        if constraints_snapped:
            return f"Tightened {constraints_snapped} constraints (snapped to equality)"

    def undo(self): ...

class SnapConstraintToEqualityDueToBounds(Transform):
    def apply(self):
        snap_count = 0
        for _, row in self.model.constraints:
            if not (row.row_type in ("L", "G")):
                continue

            should_maximize = (row.row_type == "L")

            computed = 0
            for var, coef in row.coefficients.items():
                bound = self.model.variables[var].lhs_bound if (should_maximize ^ (coef < 0)) else self.model.variables[var].rhs_bound
                if bound is None:
                    computed = None
                    break
                computed += coef * bound
            if computed is None:
                continue
            if computed == row.rhs_value:
                row.row_type = "E"
                snap_count += 1
        if snap_count:
            return f"Tightened {snap_count} constraints (snapped to equality due to bounds)"
            

class RemoveEquality(Transform):
    """Converts ax + by = c to y = c/b - a/b * x and replaces all instances of y with this expression.
    For example if another expression has 5y + 2x < 3, it will be replaced with (2 - 5*a/b)x < 3 - 5*c/b
    """

    TERM_CUTOFF = 10

    def __init__(self, model: LPModel) -> None:
        super().__init__(model)
        self.transforms = {}

    def apply(self):
        rows_deleted = 0
        while True:
            row_to_delete = None
            for name, row in self.model.constraints:
                if (
                    row.row_type == "E"
                    and len(row.coefficients) <= RemoveEquality.TERM_CUTOFF
                    and len(row.coefficients) > 1
                    and row.rhs_value == 0
                ):
                    row_to_delete = name
                    break
            if row_to_delete is None:
                break

            row = self.model.constraints_and_obj.pop(row_to_delete)
            last_var, last_coef = row.coefficients.popitem()
            assert last_coef != 0, f"Coefficient should not be zero for {last_var} in {row}"
            last_coef *= -1  # Flip to the other side
            rhs_const = -row.rhs_value / last_coef  # Flip to the other side
            coefs = {v: c / last_coef for v, c in row.coefficients.items()}

            for sub_row in self.model.constraints_and_obj.values():
                if last_var not in sub_row.coefficients:
                    continue
                scaling_coef = sub_row.coefficients.pop(last_var)
                scaled_coefs = {v: c * scaling_coef for v, c in coefs.items()}
                sub_row.rhs_value -= rhs_const * scaling_coef
                for v, c in scaled_coefs.items():
                    if v in sub_row.coefficients:
                        sub_row.coefficients[v] += c
                    else:
                        sub_row.coefficients[v] = c
                sub_row.coefficients = {v: c for v, c in sub_row.coefficients.items() if c != 0}
            self.model.constraints_and_obj = {n: r for n, r in self.model.constraints_and_obj.items() if len(r.coefficients) > 0}

            bounds = self.model.variables.pop(last_var)
            if bounds.lhs_bound is not None:
                self.model.constraints_and_obj[last_var + "__lower"] = Row(
                    last_var + "__lower",
                    "G",
                    dict(coefs),
                    rhs_value=bounds.lhs_bound - rhs_const,
                )
            if bounds.rhs_bound is not None:
                self.model.constraints_and_obj[last_var + "__upper"] = Row(
                    last_var + "__upper",
                    "L",
                    dict(coefs),
                    rhs_value=bounds.rhs_bound - rhs_const,
                )
            self.transforms[last_var] = (rhs_const, coefs, bounds)
            rows_deleted += 1
        if rows_deleted:
            return f"Removed {rows_deleted} constraints (equality w/ <{RemoveEquality.TERM_CUTOFF} terms)"

    def undo(self):
        for var, (rhs_const, coefs, bounds) in self.transforms.values():
            self.model.var_results[var] = rhs_const + sum(
                c * self.model.var_results[v] for v, c in coefs.items()
            )
            assert (
                bounds.lhs_bound is None
                or bounds.lhs_bound <= self.model.var_results[var]
            )
            assert (
                bounds.rhs_bound is None
                or bounds.rhs_bound >= self.model.var_results[var]
            )

class RemoveWeakerConstraints(Transform):
    def apply(self):
        rows_deleted = 0
        for var in self.model.variables:
            tightest_row = None
            loose_rows = []
            for row_name, row in self.model.constraints:
                if not (var in row.coefficients and row.row_type in ("L", "G")):
                    continue
                var_coef = row.coefficients[var]
                coefs = {v: -c / var_coef for v, c in row.coefficients.items() if v != var}
                # TODO it can also be pos if the variable is negative and the coef is positive
                all_pos = all(self.model.variables[v].lhs_bound == 0 and c > 0 for v, c in coefs.items())
                if not all_pos:
                    tightest_row = None
                    break
                rhs = row.rhs_value / var_coef
                # We can generalize this to allow upper bounds
                is_upper_bound = (row.row_type == "L") ^ (var_coef < 0)
                if not is_upper_bound:
                    tightest_row = None
                    break
                if tightest_row is not None:
                    tightest_rhs, tightest_coefs, _ = tightest_row
                    if not set(tightest_coefs) == set(coefs):
                        tightest_row = None
                        break
                    if rhs < tightest_rhs and all(c < tightest_coefs[v] for v, c in coefs.items()):
                        loose_rows.append(tightest_row)
                        tightest_row = (rhs, coefs, row_name)
                    else:
                        loose_rows.append((rhs, coefs, row_name))
                else:
                    tightest_row = (rhs, coefs, row_name)
            
            if tightest_row is None:
                continue

            for _, _, row_name in loose_rows:
                del self.model.constraints_and_obj[row_name]
                rows_deleted += 1
        if rows_deleted:
            return f"Removed {rows_deleted} constraints (weaker constraints)"
    
    def undo(self): ...

class MakeIntoMaximization(Transform):
    def apply(self):
        if self.model.minimization_problem:
            self.model.minimization_problem = False
            obj_row = self.model.objective
            obj_row.coefficients = {
                var: -coef for var, coef in obj_row.coefficients.items()
            }
            obj_row.rhs_value = -obj_row.rhs_value

    def undo(self): ...


class ReplaceUnboundedVariables(Transform):
    def apply(self):
        """Replaces an unbounded x1 with x1_pos and x1_neg, where x1 = x1_pos - x1_neg and x1_pos, x1_neg >= 0."""
        # Create a mapping from unbounded_vars to their x1_pos and x1_neg counterparts
        self.unbounded_vars = {
            var_name: (f"{var_name}__pos", f"{var_name}__neg")
            for var_name, bound in self.model.variables.items()
            if bound.lhs_bound is None and bound.rhs_bound is None
        }

        # Add bounds
        for pos_var, neg_var in self.unbounded_vars.values():
            self.model.variables[pos_var] = Variable(pos_var, lhs_bound=0)
            self.model.variables[neg_var] = Variable(neg_var, lhs_bound=0)

        # Replace unbounded variables with x1_pos and x1_neg in the constraints
        for var_name, (pos_var, neg_var) in self.unbounded_vars.items():
            for row in self.model.constraints_and_obj.values():
                if var_name in row.coefficients:
                    row.coefficients[pos_var] = row.coefficients[var_name]
                    row.coefficients[neg_var] = -row.coefficients[var_name]
                    del row.coefficients[var_name]

        for var_name in self.unbounded_vars:
            del self.model.variables[var_name]

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
        for var, bound in self.model.variables.items():
            if not (bound.rhs_bound is not None and bound.lhs_bound is None):
                continue
            flipped_var = f"{var}__inv"
            # Multiply variable by negative 1
            for row in self.model.constraints_and_obj.values():
                if var in row.coefficients:
                    row.coefficients[flipped_var] = -row.coefficients.pop(var)
            new_bounds[flipped_var] = Variable(flipped_var, lhs_bound=-bound.rhs_bound)
            self.flipped_vars[var] = flipped_var
        for var, flipped_var in self.flipped_vars.items():
            del self.model.variables[var]
            self.model.variables[flipped_var] = new_bounds[flipped_var]

    def undo(self):
        for var, flipped_var in self.flipped_vars.items():
            self.model.var_results[var] = -self.model.var_results[flipped_var]
            del self.model.var_results[flipped_var]


class MoveUpperBoundToConstraint(Transform):
    def apply(self):
        """Converts a<=x<=b to x>=a with constraint x<=b."""
        for var, bound in self.model.variables.items():
            if not (bound.rhs_bound is not None and bound.lhs_bound is not None):
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
            for var, bound in self.model.variables.items()
            if bound.lhs_bound is not None and bound.lhs_bound != 0
        }

        for var, (shift_var, shift_amount) in self.shifted_vars.items():
            for row in self.model.constraints_and_obj.values():
                if var in row.coefficients:
                    row.rhs_value += shift_amount * row.coefficients[var]
                    row.coefficients[shift_var] = row.coefficients[var]
                    del row.coefficients[var]

            self.model.variables[shift_var] = Variable(shift_var, lhs_bound=0)
            del self.model.variables[var]

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
        for row_name, row in self.model.constraints:
            # Ensure a positive RHS by multiplying by -1
            if row.rhs_value < 0:
                row.coefficients = {v: -c for v, c in row.coefficients.items()}
                row.rhs_value = -row.rhs_value
                if row.row_type == "L":
                    row.row_type = "G"
                elif row.row_type == "G":
                    row.row_type = "L"

            # If we have a = or >= we need to add an artificial variable
            if row.row_type in ("G"):
                artificial_name = f"art__{row_name}"
                self.artificial_vars.append(artificial_name)
                row.coefficients[artificial_name] = 1
                self.model.objective.coefficients[artificial_name] = BIG_M
                self.model.variables[artificial_name] = Variable(
                    artificial_name, lhs_bound=0
                )

            # If we have <= or >= we need to add a slack variable
            if row.row_type in ("L", "G"):
                slack_name = f"slack__{row_name}"
                self.artificial_vars.append(slack_name)
                row.coefficients[slack_name] = 1 if row.row_type == "L" else -1
                self.model.variables[slack_name] = Variable(slack_name, lhs_bound=0)
                row.row_type = "E"

    def undo(self):
        for var in self.artificial_vars:
            # TODO use these values to find the constraints' duals?
            del self.model.var_results[var]

def checkpoint(model, name: str):
    print("\n" + name.upper())
    export_to_lp_file(model, f"{name}.lp")
    print(f"Model size: ({len(model.constraints_and_obj)}, {len(model.variables)})")
    gp.setParam("LogToConsole", 0)
    m = gp.read(f"{name}.lp")
    m.Params.Method = 0
    m.Params.InfUnbdInfo = 1
    p = m.presolve()
    p.write(f"{name}_gp_presolve.lp")
    
    m.optimize()
    print(f"Objective value: {m.objVal}")
    # print(*((v.VarName, v.X) for v in m.getVars()), sep="\n")
    

def transform_into_standard_form(model):
    checkpoint(model, "before_presolve")
    presolve_transforms = MultiTryTransform(
        model,
        ConstraintToBound,
        RemoveVarsFixedByConstraint,
        RemoveVarsFixedByBound,
        SnapConstraintToEqualityDueToBounds,
        SnapConstraintsToEquality,
        RemoveEquality,
        RemoveUnusedVariables,
        RemoveEmptyConstraints,
        PushVarsToBounds,
        RemoveWeakerConstraints
    )
    presolve_transforms.apply()
    checkpoint(model, "after_presolve")
    transforms = MultiTransform(
        model,
        MoveUpperBoundToConstraint,
        FlipUpperBounds,
        ReplaceUnboundedVariables,
        ShiftBoundsToZero,
        AddArtificialVariables,
        MakeIntoMaximization,
    )
    transforms.apply()
    checkpoint(model, "after_standard_form")
    model.assert_is_in_standard_form()

def export_to_lp_file(model: LPModel, filename: str):
    model.assert_is_valid()
    with open(filename, "w") as f:
        f.write("Minimize\n") if model.minimization_problem else f.write("Maximize\n")
        f.write("obj: ")
        for var, coef in model.objective.coefficients.items():
            f.write(f"{coef:+.12g} {var} ")
        f.write(f"{(-model.objective.rhs_value + 0.):+.12g} Constant\n")

        f.write("\nSubject To\n")
        for _, row in model.constraints:
            f.write(f"{row}\n")
        f.write("\nBounds\n")
        f.write("Constant = 1\n")
        for var, bound in model.variables.items():
            # Doing + 0. to avoid -0.0, in Python 3.11 and up the format specifier z can be used (see PEP682)
            if bound.lhs_bound is None:
                lhs = "-Inf <= "
            elif bound.lhs_bound == 0:
                lhs = ""
            else:
                lhs = f"{bound.lhs_bound + 0:.12g} <= "
            rhs = "" if bound.rhs_bound is None else f" <= {bound.rhs_bound + 0:.12g}"
            if lhs or rhs:
                f.write(f"{lhs}{var}{rhs}\n")
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
        shape=(len(model.constraints_and_obj), len(model.variables)),
        dtype=np.float64,
    ).toarray()
