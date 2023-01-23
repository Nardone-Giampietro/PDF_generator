import unittest
from scipy import stats
from splrandom import ProbabilityDensityFunction
import numpy as np

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
