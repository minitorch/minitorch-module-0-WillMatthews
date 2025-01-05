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
    return max(x,y)

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

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

#def negList(ls: Iterable[float]) -> Iterable[float]:


# TODO: Implement for Task 0.3.
