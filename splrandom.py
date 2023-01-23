"""
Created by Nardone Giampietro with GPL-3.0 license
"""
import logging
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """
    This class creates an istance that inherit from the scipy_ class InterpolatedUnivariateSpline_\
    The instance accept two arrays sampling the PDF on a grid of values.

    .. _scipy: https://docs.scipy.org/doc/scipy/
    .. _InterpolatedUnivariateSpline: \
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

    :param x_array: 1-D array of values where the PDF has been sampled over. Must be increasing
    :type x_array: array[float]
    :param y_array: 1-D array of the normalized occurrences for each sampled value.
    :type y_array: array[float]
    :param k: Degree of the smoothing spline used to interpolate the input data.\
        Must be 1 <= k <= 5. k = 3 is a cubic spline. Default is 3.
    :type k: integer or None

    """

    def __init__(self, x_array, y_array, k=3):
        """
        Initialize the ProbabilityDensityFunction istance.
        """
        InterpolatedUnivariateSpline.__init__(self, x_array, y_array, k=k)
        self._x0 = x_array[0]
        self._xf = x_array[len(x_array)-1]
        y_cdf = np.array([self.integral(x_array[0], x_val) for x_val in x_array])
        self.cdf = InterpolatedUnivariateSpline(x_array, y_cdf)
        x_idf, index = np.unique(y_cdf, return_index=True)
        y_idf = x_array[index]
        self.idf = InterpolatedUnivariateSpline(x_idf, y_idf)

    def int_rand(self, inter):
        """
        :param inter: Interval of interest.
        :type inter: list[float, float]
        :return: The probability of finding a random generated value inside the interval.
        :rtype: float
        """
        if len(inter) != 2:
            logging.error("Invalid interval. It must be an array of shape [2]")
            sys.exit()
        elif (inter[0] < self._x0) or (inter[1] > self._xf):
            logging.error(
                "Invalid interval. It must be included inside\
                [%.2f, %.2f].", self._x0, self._xf
            )
            sys.exit()
        else:
            return self.cdf(inter[1]) - self.cdf(inter[0])

    def rand(self, size=1):
        """
        :param size: Number of pseudo-random values to be generated\
            with the sampled PDF. Default is 1.
        :type size: int
        :return: List of pseudo-random numbers.
        :rtype: list[float]
        """
        return self.idf(np.random.uniform(size=size))
