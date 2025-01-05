"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


def mul (x: float, y: float) -> float:
    """
    Multiplication

    Args:
        x: float
        y: float

    Returns:
        x*y
    """
    return x * y

def id(x:float) -> float:
    """
    Identity function

    Args:
        x: float

    Returns:
        The same as the input
    """

    return x

def add (x: float, y: float) -> float:
    """
    Addition

    Args:
        x: float
        y: float

    Returns:
        The sum of x and y
    """
    return x+y

def neg(x: float) -> float:
    """
    Negation

    Args:
        x: float

    Returns:
        The negation of x
    """
    return -x

def lt(x:float, y:float) -> bool:
    """
    Less than test

    Args:
        x: float
        y: float

    Returns:
        True if x is less than y, False otherwise
    """
    return x < y

def eq(x:float, y:float) -> bool:
    """
    Equality test

    Args:
        x: float
        y: float

    Returns:
        True if x and y are equal, False otherwise
    """
    return x == y

def max(x:float, y:float) -> float:
    """
    Maximum of two numbers

    Args:
        x: float
        y: float

    Returns:
        The maximum of x and y
    """
    if x > y:
        return x
    return y

def is_close (x:float,y:float) -> bool:
    """
    Determine if two floating-point numbers are close to each other within a small threshold.

    Args:
        x (float): The first floating-point number.
        y (float): The second floating-point number.

    Returns:
        bool: True if the absolute difference between x and y is less than 1e-2, False otherwise.
    """
    thold = 1e-2
    mag = abs(add(x,neg(y)))
    return lt(mag, thold)

def sigmoid(x:float) -> float:
    """
    Compute the sigmoid function for the input value x.

    The sigmoid function is defined as:
    - f(x) = 1 / (1 + exp(-x)) if x >= 0
    - f(x) = exp(x) / (1 + exp(x)) otherwise

    Args:
        x (float): The input value.

    Returns:
        float: The computed sigmoid value.
    """
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        return 1.0/(1.0  + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0+ex)

def relu(x: float) ->  float:
    """
    Compute the ReLU function for the input value x.

    The ReLU function is defined as:
    - f(x) = x if x > 0
    - f(x) = 0 otherwise

    Args:
        x (float): The input value.

    Returns:
        float: The computed ReLU value.
    """

    if x >= 0:
        return x
    return 0

def log(x:float) -> float:
    """
    Compute the natural logarithm of the input value x.

    Args:
        x (float): The input value.

    Returns:
        float: The computed natural logarithm value.
    """
    return math.log(x)

def exp(x:float) -> float:
    """
    Compute the exponential of the input value x.

    Args:
        x (float): The input value.

    Returns:
        float: The computed exponential value.
    """
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """
    Compute the derivative of the natural logarithm function at x.

    The derivative of the natural logarithm function is defined as:
    - f'(x) = 1/x

    Args:
        x (float): The input value.
        y (float): A value that is multiplied by the derivative.s

    Returns:
        float: The computed gradient.
    """

    deriv = 1/x
    return y * deriv

def inv(x:float) -> float:
    """
    Computes the reciprocal of the input value x.

    Args:
        x (float): The input value.

    Returns:
        float: The computed reciprocal value.
    """
    return 1/x

def inv_back(x: float, y: float) -> float:
    """
    Compute the derivative of the reciprocal function at x.

    The derivative of the reciprocal function is defined as:
    - f'(x) = -1/x^2

    Args:
        x (float): The input value.
        y (float): A value that is multiplied by the derivative.

    Returns:
        float: The computed gradient.
    """
    deriv = - inv(x)**2
    return y * deriv


def relu_back(x:float, y:float) -> float:
    """
    Compute the derivative of the ReLU function at x.

    The derivative of the ReLU function is defined as:
    - f'(x) = 1 if x > 0
    - f'(x) = 0 otherwise

    Args:
        x (float): The input value.
        y (float): A value that is multiplied by the derivative.

    Returns:
        float: The computed gradient.
    """
    if x > 0:
        return y
    return 0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """
    Map a function over a list.

    Args:
        fn (Callable[[float], float]): The function to map over the list.
        ls (Iterable[float]): The list to map the function over.

    Returns:
        Iterable[float]: The list with the function mapped over it.
    """
    return [fn(x) for x in ls]


def zipWith(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Zip two lists together with a function.

    Args:
        fn (Callable[[float, float], float]): The function to zip the lists with.
        ls1 (Iterable[float]): The first list to zip.
        ls2 (Iterable[float]): The second list to zip.

    Returns:
        Iterable[float]: The list zipped with the function.
    """
    return [fn(x,y) for x,y in zip(ls1,ls2)]

def reduce(fn: Callable[[float, float], float], ls: Iterable[float], init: float) -> float:
    """
    Reduce a list with a function and an initial value.

    Args:
        fn (Callable[[float, float], float]): The function to reduce the list with.
        ls (Iterable[float]): The list to reduce.
        init (float): The initial value to start the reduction.

    Returns:
        float: The reduced value.
    """
    acc = init
    for x in ls:
        acc = fn(acc,x)
    return acc


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate a list.

    Args:
        ls (Iterable[float]): The list to negate.

    Returns:
        Iterable[float]: The negated list.
    """
    return map(neg,ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add two lists together.

    Args:
        ls1 (Iterable[float]): The first list to add.
        ls2 (Iterable[float]): The second list to add.

    Returns:
        Iterable[float]: The sum of the two lists.
    """
    return zipWith(add,ls1,ls2)


def sum(ls: Iterable[float]) -> float:
    """
    Sum a list of numbers.

    Args:
        ls (Iterable[float]): The list to sum.

    Returns:
        float: The sum of the list.
    """
    return reduce(add,ls,0)

def prod(ls: Iterable[float]) -> float:
    """
    Product of a list of numbers.

    Args:
        ls (Iterable[float]): The list to multiply.

    Returns:
        float: The product of the list.
    """
    return reduce(mul, ls, 1)
