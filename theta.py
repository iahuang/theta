"""
theta.py - A runtime complexity analysis library written in Python.

Released under the MIT License
"""

import random
import timeit
import math
from dataclasses import dataclass
from typing import Any, Iterable, Union, Callable


class InputSizeVariable:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __add__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "+")

    def __sub__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "-")
    
    def __mul__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "*")
    
    def __div__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "/")
    
    def __pow__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "**")

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, to: "InputSizeVariable") -> bool:
        return to.name == self.name

    def __repr__(self) -> str:
        return '<Variable "{}">'.format(self.name)


class _Expression:
    def __add__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "+")

    def __sub__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "-")
    
    def __mul__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "*")
    
    def __div__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "/")
    
    def __pow__(self, other: "_ExprEvaluable") -> "_Expression":
        return _ExprBinaryOp(self, other, "**")


class _ExprBinaryOp(_Expression):
    a: "_ExprEvaluable"
    b: "_ExprEvaluable"
    op: str

    def __init__(self, a: "_ExprEvaluable", b: "_ExprEvaluable", op: str):
        self.a = a
        self.b = b
        self.op = op

class _ExprLog(_Expression):
    x: "_ExprEvaluable"

    def __init__(self, x: "_ExprEvaluable"):
        self.x = x

def Log(x: "_ExprEvaluable"):
    return _ExprLog(x)

_ExprEvaluable = Union[_Expression, InputSizeVariable, float, int]


def _evaluate(node: _ExprEvaluable, var_values: dict[InputSizeVariable, int]) -> float:
    if isinstance(node, InputSizeVariable):
        if not node in var_values:
            raise NameError("Variable '{}' is not defined".format(node.name))

        return float(var_values[node])
    
    elif isinstance(node, _ExprLog):
        return math.log(_evaluate(node.x, var_values))

    elif isinstance(node, _ExprBinaryOp):
        a_value = _evaluate(node.a, var_values)
        b_value = _evaluate(node.b, var_values)

        if node.op == "+":
            return a_value + b_value
        elif node.op == "-":
            return a_value - b_value
        elif node.op == "*":
            return a_value * b_value
        elif node.op == "**":
            return a_value ** b_value
        elif node.op == "/":
            if b_value == 0:
                return math.inf

            return a_value / b_value
        else:
            raise ArithmeticError("Unsupported operation '{}'".format(node.op))
    elif isinstance(node, int) or isinstance(node, float):
        return float(node)
    else:
        raise TypeError


@dataclass
class RuntimeDataPoint:
    """
    We can think of this class as a point on "f", the function represented by the plotted
    runtime data for a chosen piece of code. In this case, self.input_sizes represents the
    arguments to "f" and self.exec_time represents "f(x)". 
    """
    input_sizes: dict[InputSizeVariable, int]
    exec_time: float


class RuntimeData:
    _points: list[RuntimeDataPoint]

    def __init__(self):
        self._points = []

    def add_data_point(self, point: RuntimeDataPoint) -> None:
        self._points.append(point)

    def get_points(self) -> list[RuntimeDataPoint]:
        return [p for p in self._points]

    def size(self) -> int:
        return len(self._points)


@dataclass
class FunctionInput:
    args: list[Any]
    input_sizes: dict[InputSizeVariable, int]


def compile_runtime_data(
    f: Callable,
    function_inputs: Iterable[FunctionInput],
    min_iters: int,
    target_time_per_input: float = 1.0,
) -> RuntimeData:
    """
    Take a function "f" and a list of sample inputs "function_inputs". Measure the
    average execution time of the function for each of the inputs and return a RuntimeData object.

    Whenever possible, the function's execution time will be measured at least "min_iters" times,
    but will stop after measuring for "target_time_per_input" seconds.

    Due to the potentially very small timescale of "f", an optional argument "n"
    may be specified to count each execution of "f" as the sum of its execution "n" times.
    """

    runtime_data = RuntimeData()

    for f_input in function_inputs:
        total_iters = min_iters
        total_time = 0
        def _run_func():
            f(*f_input.args)

        # start by trying to run the function "min_iters" times and measure the runtime
        diagnostic_exec_time = timeit.timeit(_run_func, number=min_iters)
        total_time += diagnostic_exec_time

        # if the time it took to do that did not exceed target_time_per_input,
        # estimate how many more iterations we can run, and perform more measurements

        est_time_per_iter = diagnostic_exec_time / min_iters
        remaining_time = target_time_per_input - diagnostic_exec_time

        if remaining_time > 0:
            extra_iters = round(remaining_time / est_time_per_iter)
            total_iters += extra_iters

            # measure the function runtime again
            total_time += timeit.timeit(_run_func, number=extra_iters)

        # measured in time (seconds) per function call
        avg_exec_time = total_time / total_iters

        # add to runtime_data
        runtime_data.add_data_point(RuntimeDataPoint(
            input_sizes=f_input.input_sizes,
            exec_time=avg_exec_time
        ))

    return runtime_data


def _mse(f: RuntimeData, g: _ExprEvaluable, c: float) -> float:
    sum_sq_err = 0

    for point in f.get_points():
        err = point.exec_time - _evaluate(g, point.input_sizes) * c
        sum_sq_err += err*err

    return sum_sq_err / f.size()

def _regress_c(f: RuntimeData, g: _ExprEvaluable) -> tuple[float, float]:
    """
    We define f(a,b,c...) in O(g(a,b,c,...)) as there existing some n > 0 and c > 0 such that
    for all a,b,c,... > 0, c*g(a,b,c,...) > f(a,b,c,...) whenever a,b,c,... > n

    The goal of this function is to fit a value for "c" and return a MSE loss for "g"
    to the time complexity function "f" represented by plotted runtime data points.

    Return the tuple (c, MSE)
    """
    c = 0

    # Find an initial estimate for c based on approximating the limit of
    # f(a,b,c,...) / g(a,b,c,...) as the vector <a,b,c,...> approaches infinity.

    # estimate the limit of f/g by sampling the "furthest" out point from zero.
    max_dist = 0.0
    limit = 0.0

    for point in f.get_points():
        magnitude = sum(
            component*component for component in point.input_sizes.values())

        if magnitude > max_dist:
            limit = point.exec_time / _evaluate(g, point.input_sizes)
            max_dist = magnitude

    c = limit
    mse = 0.0

    # estimate a value for epsilon based on the magnitude of our estimate for c.
    _m = round(math.log10(c))
    EPSILON = math.pow(10, _m-2)
    ITERS = 200

    # continuously try values of c +/- epsilon to minimize MSE. This method is inefficient, but
    # should work.

    for _ in range(ITERS):
        loss_c_plus = _mse(f, g, c+EPSILON)
        loss_c_minus = _mse(f, g, c-EPSILON)

        mse = min(loss_c_plus, loss_c_minus)

        if loss_c_minus < loss_c_plus:
            c -= EPSILON
        elif loss_c_plus < loss_c_minus:
            c += EPSILON
        else:
            break
            
        # print({
        #     "c": c,
        #     "loss": mse
        # })

    return (c, mse)

def bigO_correlation(data: RuntimeData, complexity_func: _ExprEvaluable):
    _, mse = _regress_c(data, complexity_func)

    return math.log(1/mse)

if __name__ == "__main__":
    N = InputSizeVariable("n")
    M = InputSizeVariable("m")

    def test_function(x: list[int], y: list[int]):
        lsum = 0

        for a in x:
            for b in y:
                lsum += a+b

        return lsum

    input_generator = (
        FunctionInput(
            args=[[random.randint(0, 10) for _ in range(i1)], [random.randint(0, 10) for _ in range(i2)]],
            input_sizes={
                N: i1,
                M: i2
            }
        )
        for i1 in [5, 10, 20, 40, 80, 160, 320, 640]
        for i2 in [5, 10, 20, 40, 80, 160, 320, 640]
    )

    data = compile_runtime_data(
        f=test_function,
        function_inputs=input_generator,
        min_iters=200,
    )

    print("O(n)     ", bigO_correlation(data, N))
    print("O(m)     ", bigO_correlation(data, M))
    print("O(nm)    ", bigO_correlation(data, N*M))
    print("O(nlogm) ", bigO_correlation(data, N*Log(M)))
    