"""Code inspired from pyomo/pyomo library. Copyright is therefore theirs."""

import os
import re
import time
import ece532_solver

from pyomo.opt.base.formats import ProblemFormat, ResultsFormat
from pyomo.opt.solver.ilmcmd import ILMLicensedSystemCallSolver
from pyomo.common.collections.bunch import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.results import (
    Solution,
    TerminationCondition,
    SolutionStatus,
    SolverStatus,
    ProblemSense,
)
from pyomo.opt.base.solvers import SolverFactory


@SolverFactory.register(
    "ece532_solver",
    doc="Custom solver that reads from an MPS file and writes to a solution file",
)
class ECE532Solver(ILMLicensedSystemCallSolver):
    """Shell interface to the GUROBI LP/MIP solver"""

    _solver_info_cache = {}

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds["type"] = "custom"
        ILMLicensedSystemCallSolver.__init__(self, **kwds)

        # NOTE: eventually both of the following attributes should be
        # migrated to a common base class.  is the current solve
        # warm-started? a transient data member to communicate state
        # information across the _presolve, _apply_solver, and
        # _postsolve methods.
        self._warm_start_solve = False
        # related to the above, the temporary name of the MST warm-start
        # file (if any).
        self._warm_start_file_name = None

        #
        # Define valid problem formats and associated results formats
        #
        self._valid_problem_formats = [ProblemFormat.mps]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.mps)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = False
        self._capabilities.quadratic_constraint = False
        self._capabilities.integer = False
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    def license_is_valid(self):
        return True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def warm_start_capable(self):
        return False

    def _default_executable(self):
        return "python"

    def _execute_command(self, command):
        """
        Execute the command
        """

        start_time = time.time()

        ece532_solver.main_without_argument_parser(
            self._problem_files[0], output_file=self._soln_file
        )

        self._last_solve_time = time.time() - start_time

        return [0, ""]

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        return (0, 0, 1, 0)

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        # The log file in CPLEX contains the solution trace, but the
        # solver status can be found in the solution file.
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix=".ece532.log")

        #
        # Define solution file
        # As indicated above, contains (in XML) both the solution and
        # solver status.
        #
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix=".ece532.txt")

        #
        # Write the GUROBI execution script
        #

        problem_filename = self._problem_files[0]
        solution_filename = self._soln_file

        # translate the options into a normal python dictionary, from a
        # pyutilib SectionWrapper - the gurobi_run function doesn't know
        # about pyomo, so the translation is necessary.
        options_dict = {}
        for key in self.options:
            options_dict[key] = self.options[key]

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, script="", log_file=self._log_file, env=None)

    def process_soln_file(self, results):
        # the only suffixes that we extract from CPLEX are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_rc = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_rc = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The GUROBI solver plugin cannot extract solution suffix="
                    + suffix
                )

        # check for existence of the solution file
        # not sure why we just return - would think that we
        # would want to indicate some sort of error
        if not os.path.exists(self._soln_file):
            return

        soln = Solution()

        # caching for efficiency
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        num_variables_read = 0

        # string compares are too expensive, so simply introduce some
        # section IDs.
        # 0 - unknown
        # 1 - problem
        # 2 - solution
        # 3 - solver

        section = 0  # unknown

        solution_seen = False

        range_duals = {}
        range_slacks = {}

        INPUT = open(self._soln_file, "r")
        for line in INPUT:
            line = line.strip()
            tokens = [token.strip() for token in line.split(":")]
            if tokens[0] == "section":
                if tokens[1] == "problem":
                    section = 1
                elif tokens[1] == "solution":
                    section = 2
                    solution_seen = True
                elif tokens[1] == "solver":
                    section = 3
            else:
                if section == 2:
                    if tokens[0] == "var":
                        if tokens[1] != "ONE_VAR_CONSTANT":
                            soln_variables[tokens[1]] = {"Value": float(tokens[2])}
                            num_variables_read += 1
                    elif tokens[0] == "status":
                        soln.status = getattr(SolutionStatus, tokens[1])
                    elif tokens[0] == "gap":
                        soln.gap = float(tokens[1])
                    elif tokens[0] == "objective":
                        if tokens[1].strip() != "None":
                            soln.objective["__default_objective__"] = {
                                "Value": float(tokens[1])
                            }
                            if results.problem.sense == ProblemSense.minimize:
                                results.problem.upper_bound = float(tokens[1])
                            else:
                                results.problem.lower_bound = float(tokens[1])
                    elif tokens[0] == "constraintdual":
                        name = tokens[1]
                        if name != "c_e_ONE_VAR_CONSTANT":
                            if name.startswith("c_"):
                                soln_constraints.setdefault(tokens[1], {})["Dual"] = (
                                    float(tokens[2])
                                )
                            elif name.startswith("r_l_"):
                                range_duals.setdefault(name[4:], [0, 0])[0] = float(
                                    tokens[2]
                                )
                            elif name.startswith("r_u_"):
                                range_duals.setdefault(name[4:], [0, 0])[1] = float(
                                    tokens[2]
                                )
                    elif tokens[0] == "constraintslack":
                        name = tokens[1]
                        if name != "c_e_ONE_VAR_CONSTANT":
                            if name.startswith("c_"):
                                soln_constraints.setdefault(tokens[1], {})["Slack"] = (
                                    float(tokens[2])
                                )
                            elif name.startswith("r_l_"):
                                range_slacks.setdefault(name[4:], [0, 0])[0] = float(
                                    tokens[2]
                                )
                            elif name.startswith("r_u_"):
                                range_slacks.setdefault(name[4:], [0, 0])[1] = float(
                                    tokens[2]
                                )
                    elif tokens[0] == "varrc":
                        if tokens[1] != "ONE_VAR_CONSTANT":
                            soln_variables[tokens[1]]["Rc"] = float(tokens[2])
                    else:
                        setattr(soln, tokens[0], tokens[1])
                elif section == 1:
                    if tokens[0] == "sense":
                        if tokens[1] == "minimize":
                            results.problem.sense = ProblemSense.minimize
                        elif tokens[1] == "maximize":
                            results.problem.sense = ProblemSense.maximize
                    else:
                        try:
                            val = eval(tokens[1])
                        except:
                            val = tokens[1]
                        setattr(results.problem, tokens[0], val)
                elif section == 3:
                    if tokens[0] == "status":
                        results.solver.status = getattr(SolverStatus, tokens[1])
                    elif tokens[0] == "termination_condition":
                        try:
                            results.solver.termination_condition = getattr(
                                TerminationCondition, tokens[1]
                            )
                        except AttributeError:
                            results.solver.termination_condition = (
                                TerminationCondition.unknown
                            )
                    else:
                        setattr(results.solver, tokens[0], tokens[1])

        INPUT.close()

        # For the range constraints, supply only the dual with the largest
        # magnitude (at least one should always be numerically zero)
        for key, (ld, ud) in range_duals.items():
            if abs(ld) > abs(ud):
                soln_constraints["r_l_" + key] = {"Dual": ld}
            else:
                # Use the same key
                soln_constraints["r_l_" + key] = {"Dual": ud}
        # slacks
        for key, (ls, us) in range_slacks.items():
            if abs(ls) > abs(us):
                soln_constraints.setdefault("r_l_" + key, {})["Slack"] = ls
            else:
                # Use the same key
                soln_constraints.setdefault("r_l_" + key, {})["Slack"] = us

        if solution_seen:
            results.solution.insert(soln)
