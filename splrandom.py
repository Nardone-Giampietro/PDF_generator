"""
Created by Nardone Giampietro with GPL-3.0 license
"""
import logging
import sys
import unittest
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
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

class TestProbabilityDensityFunction(unittest.TestCase):
    """
    Test of the class ProbabilityDensityFunction with differents PDFs using the \
    Kolmogorov-Smirnov test for goodness of fit.
    """
    c_triang = 0.5
    loc_triang = 0.0
    scale_triang = 1.0
    scale_norm = 1.0
    loc_norm = 0.0

    def triang_cdf(self, x_var):
        """
        Cumulant Distribution Function of the triangular pdf
        """
        return stats.triang.cdf(x_var, self.c_triang, loc=self.loc_triang, scale=self.scale_triang)

    def norm_cdf(self, x_var):
        """
        Cumulant Distribution Function of the normal pdf
        """
        return stats.norm.cdf(x_var, loc=self.loc_norm, scale=self.scale_norm)

    def test_triangular_pvalue(self):
        """
        Testing with the triangular pdf
        """
        x_spl = np.linspace(self.loc_triang, self.scale_triang, 50)
        y_spl = np.array([stats.triang.pdf(val,self.c_triang) for val in x_spl])
        spl_pdf_triangular = ProbabilityDensityFunction(x_spl, y_spl)
        ks_test_result = stats.kstest(spl_pdf_triangular.rand(size=1000), self.triang_cdf)
        p_value = ks_test_result.pvalue
        self.assertTrue(p_value >= 0.05, msg="Triangular PDF test NOT passed")

    def test_norm_pvalue(self):
        """
        Testing with the normal pdf
        """
        x_spl = np.linspace(-5.0, 5.0, 50)
        y_spl = np.array(
                [stats.norm.pdf(val, loc=self.loc_norm, scale=self.scale_norm) for val in x_spl]
                )
        spl_pdf_norm = ProbabilityDensityFunction(x_spl, y_spl)
        ks_test_result = stats.kstest(spl_pdf_norm.rand(size=1000), self.norm_cdf)
        p_value = ks_test_result.pvalue
        self.assertTrue(p_value >= 0.05, msg="Normal PDF test NOT passed")

if __name__ == "__main__":
    unittest.main()
