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
    return -x

def lt(x:float, y:float) -> bool:
    return x < y

def eq(x:float, y:float) -> bool:
    return x == y

def max(x:float, y:float) -> float:
    return max(x,y)

def is_close (x:float,y:float) -> bool:
    # $f(x) = |x - y| < 1e-2$
    thold = 1e-2
    mag = abs(add(x,neg(y)))
    return lt(mag, thold)

def sigmoid(x:float) -> float:
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        return 1.0/(1.0  + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0+ex)

def relu(x: float) ->  float:
    if x >= 0:
        return x
    return 0

def log(x:float) -> float:
    return math.log(x)

def exp(x:float) -> float:
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """derivative of log times a second arg"""
    deriv = 1/x
    return y * deriv

def inv(x:float) -> float:
    return 1/x

# - inv_back
def inv_back(x: float, y: float) -> float:
    deriv = - inv(x)**2
    return y * deriv


# - relu_back
def relu_back(x:float, y:float) -> float:
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
