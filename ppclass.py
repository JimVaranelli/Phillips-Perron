import sys
import os
import time
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from numpy.testing import assert_equal, assert_almost_equal


class PhillipsPerron(object):
    """
    Class wrapper for Phillips-Perron unit-root test
    """

    def __init__(self):
        """
        Critical values for the Z-rho statistic under three different models
        specified for the Phillips-Perron unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        1,000,000 replications and 5000 data points (see code file cvmp.py).
        """
        self.__pp_critical_values = {}
        # no constant, no trend
        self.__nc = (
            (0.0001, -43.843451), (1.0000, -13.403079), (4.0000, -8.636551),
            (5.0000, -7.894793), (8.0000, -6.325709), (10.0000, -5.601655),
            (12.0000, -5.019618), (14.0000, -4.534404), (16.0000, -4.122150),
            (18.0000, -3.759609), (20.0000, -3.437565), (22.0000, -3.148533),
            (24.0000, -2.890398), (26.0000, -2.656496), (28.0000, -2.440494),
            (30.0000, -2.242433), (32.0000, -2.058722), (34.0000, -1.887868),
            (36.0000, -1.726873), (38.0000, -1.576115), (40.0000, -1.433285),
            (42.0000, -1.300564), (44.0000, -1.176476), (46.0000, -1.059077),
            (48.0000, -0.944997), (50.0000, -0.836941), (52.0000, -0.734356),
            (54.0000, -0.634711), (56.0000, -0.537629), (58.0000, -0.444437),
            (60.0000, -0.354470), (62.0000, -0.265406), (64.0000, -0.178867),
            (66.0000, -0.095355), (68.0000, -0.012679), (70.0000, 0.067170),
            (72.0000, 0.145965), (74.0000, 0.225055), (76.0000, 0.303231),
            (78.0000, 0.380918), (80.0000, 0.460492), (82.0000, 0.540384),
            (84.0000, 0.623234), (86.0000, 0.711703), (88.0000, 0.805489),
            (90.0000, 0.910168), (92.0000, 1.028732), (94.0000, 1.174803),
            (96.0000, 1.368292), (98.0000, 1.683951), (99.9999, 6.098997)
            )
        self.__pp_critical_values['nc'] = np.asarray(self.__nc)
        # constant-only
        self.__c = (
            (0.0001, -59.581944), (1.0000, -20.613019), (4.0000, -14.995858),
            (5.0000, -14.085536), (8.0000, -12.169208), (10.0000, -11.256604),
            (12.0000, -10.507944), (14.0000, -9.864578), (16.0000, -9.310859),
            (18.0000, -8.816190), (20.0000, -8.370138), (22.0000, -7.967712),
            (24.0000, -7.593471), (26.0000, -7.253667), (28.0000, -6.929329),
            (30.0000, -6.634397), (32.0000, -6.354931), (34.0000, -6.085966),
            (36.0000, -5.832845), (38.0000, -5.595575), (40.0000, -5.366526),
            (42.0000, -5.148250), (44.0000, -4.940322), (46.0000, -4.738416),
            (48.0000, -4.544279), (50.0000, -4.355025), (52.0000, -4.170686),
            (54.0000, -3.990399), (56.0000, -3.813245), (58.0000, -3.640756),
            (60.0000, -3.470896), (62.0000, -3.301756), (64.0000, -3.139242),
            (66.0000, -2.974775), (68.0000, -2.811561), (70.0000, -2.648991),
            (72.0000, -2.487300), (74.0000, -2.325388), (76.0000, -2.162519),
            (78.0000, -1.994356), (80.0000, -1.823856), (82.0000, -1.645439),
            (84.0000, -1.460934), (86.0000, -1.269964), (88.0000, -1.065718),
            (90.0000, -0.845276), (92.0000, -0.597655), (94.0000, -0.307744),
            (96.0000, 0.04999), (98.0000, 0.589706), (99.9999, 6.070898)
            )
        self.__pp_critical_values['c'] = np.asarray(self.__c)
        # constant + trend
        self.__ct = (
            (0.0001, -67.137465), (1.0000, -29.292846), (4.0000, -22.702592),
            (5.0000, -21.614325), (8.0000, -19.298375), (10.0000, -18.186005),
            (12.0000, -17.257239), (14.0000, -16.453414), (16.000, -15.756262),
            (18.0000, -15.127866), (20.0000, -14.557995), (22.000, -14.039898),
            (24.0000, -13.548132), (26.0000, -13.096870), (28.000, -12.677287),
            (30.0000, -12.275719), (32.0000, -11.900334), (34.000, -11.543294),
            (36.0000, -11.194222), (38.0000, -10.860353), (40.000, -10.539842),
            (42.0000, -10.229958), (44.0000, -9.931212), (46.0000, -9.645850),
            (48.0000, -9.366733), (50.0000, -9.090804), (52.0000, -8.820162),
            (54.0000, -8.552001), (56.0000, -8.289483), (58.0000, -8.033145),
            (60.0000, -7.780639), (62.0000, -7.534730), (64.0000, -7.284789),
            (66.0000, -7.038556), (68.0000, -6.791935), (70.0000, -6.545150),
            (72.0000, -6.300007), (74.0000, -6.049883), (76.0000, -5.797517),
            (78.0000, -5.537882), (80.0000, -5.275198), (82.0000, -5.002253),
            (84.0000, -4.719960), (86.0000, -4.421428), (88.0000, -4.102567),
            (90.0000, -3.763039), (92.0000, -3.378000), (94.0000, -2.925997),
            (96.0000, -2.368167), (98.0000, -1.547030), (99.9999, 5.433274)
            )
        self.__pp_critical_values['ct'] = np.asarray(self.__ct)

    def __pp_crit(self, stat, model='c'):
        """
        Linear interpolation for Phillips-Perron Z-rho p-values and critical
        values

        Parameters
        ----------
        stat : float
            The PP Z-rho test statistic
        model : {'nc','c','ct'}
            The model used when computing the statistic. 'c' is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated PP test statistic distribution
        """
        table = self.__pp_critical_values[model]
        y = table[:, 0]
        x = table[:, 1]
        # PP cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, y, x)
        cvdict = {"1%": crit_value[0], "5%": crit_value[1],
                  "10%": crit_value[2]}
        return pvalue, cvdict

    def _sigma_est_pp(self, resids, nobs, bw):
        """
        Return the Newey-West (1994) residual variances and covariances of the
        auxiliary regression utilizing the Bartlett kernel for subdiagonal
        weighting.
        """
        var = np.sum(resids**2)
        cov = 0
        for i in range(1, bw):
            resids_prod = np.dot(resids[i:], resids[:nobs - i])
            cov += 2 * resids_prod * (1. - (i / bw))
        return var / nobs, cov / nobs

    def run(self, x, regression='c', trunclag=None):
        """
        Phillips-Perron unit-root test

        The Phillips-Perron test (1998) can be used to test for a unit root in
        a univariate process.

        Parameters
        ----------
        x : array_like, 1d
            data series
        regression : {'nc', 'c', 'ct'}
            constant and trend order to include in regression
            * 'nc' : no constant, no trend
            * 'c'  : constant only (default)
            * 'ct' : constant and trend
        trunclag : {int, None}
            number of truncation lags, default=int(sqrt(nobs)/5) (SAS, 2015)

        Returns
        -------
        tau : float
            Z-tau test statistic
        tau_pv : float
            Z-tau p-value based on MacKinnon (1994) regression surface model
        lags : int
            number of truncation lags used for covariance estimation
        nobs : int
            number of observations used in regression
        tau_cvdict : dict
            critical values for the Z-tau test statistic at 1%, 5%, 10%
        rho : float
            Z-rho test statistic
        rho_pv : float
            Z-rho p-value based on interpolated simulation-derived critical
            values
        rho_cvdict : dict
            critical values for the Z-rho test statistic at 1%, 5%, 10%

        Notes
        -----
        H0 = series has a unit root (i.e., non-stationary)

        Basic process is to fit the time series under test with an AR(1) model
        using heteroscedasticity- and autocorrelation-consistent residual
        covariance estimation in order to generate the Phillips-Perron Z-rho
        and Z-tau statistics (1988) which are asymptotically equivalent to the
        Dickey-Fuller (1979, 1981) rho/tau statistics. Z-tau p-values are
        calculated using the statsmodel implementation of MacKinnon's (1994,
        2010) regression surface model. Z-rho p-values are interpolated from
        Monte-Carlo derived critical values.

        References
        ----------
        Dickey, D.A., and Fuller, W.A. (1979). Distribution of the estimators
        for autoregressive time series with a unit root. Journal of the
        American Statistical Association, 74: 427-431.

        Dickey, D.A., and Fuller, W.A. (1981). Likelihood ratio statistics for
        autoregressive time series with a unit root. Econometrica, 49:
        1057-1072.

        MacKinnon, J.G. (1994). Approximate asymptotic distribution functions
        for unit-root and cointegration tests. Journal of Business and Economic
        Statistics, 12: 167-176.

        MacKinnon, J.G. (2010). Critical values for cointegration tests.
        Working Paper 1227, Queen's University, Department of Economics.
        Retrieved from
        URL: https://www.econ.queensu.ca/research/working-papers.

        Newey, W.K., and West, K.D. (1994). A simple, positive semi-definite,
        heteroscedasticity- and autocorrelation-consistent covariance matrix.
        Econometrica, 20: 73-103.

        Phillips, P.C.B, and Perron, P. (1988). Testing for a unit root in time
        series regression. Biometrika, 75: 335-346.

        SAS Institute Inc. (2015). SAS/ETS 14.1 User's Guide. Cary, NC: SAS
        Institute Inc.

        Schwert, G.W. (1987). Effects of model specification on tests for unit
        roots in macroeconomic data. Journal of Monetary Economics, 20: 73-103.

        Seabold, S., and Perktold, J. (2010). Statsmodels: econometric and
        statistical modeling with python. In S. van der Walt and J. Millman
        (Eds.), Proceedings of the 9th Python in Science Conference (pp.
        57-61).
        """
        if regression not in ['nc', 'c', 'ct']:
            raise ValueError(
                'PP: regression option \'{}\' not understood'.format(
                    regression))
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError(
                'PP: x must be a 1d array or a 2d array with a single column')
        nobs = x.shape[0] - 1
        if trunclag is not None and (
                                trunclag < 0 or trunclag > x.shape[0] - 2):
            raise ValueError(
                'PP: trunclags must be in range [0, {}]'.format(x.shape - 2))
        # set up exog matrix
        x = np.reshape(x, (-1, 1))
        if regression == 'nc':
            exog = np.ones(shape=(nobs, 1))
        elif regression == 'c':
            exog = np.ones(shape=(nobs, 2))
        else:
            exog = np.ones(shape=(x.shape[0] - 1, 3))
            exog[:, 1] = np.arange(1, nobs + 1).reshape(nobs,)
        exog[:, -1] = x[:-1].reshape(nobs,)
        # set up endog vector
        endog = x[1:].reshape(nobs, 1)
        # run auxiliary regression
        ols = OLS(endog, exog).fit()
        # save the coefficient of the AR(1) term
        rho_hat = ols.params[-1]
        # save the standard error of the AR(1) term
        se = ols.bse[-1]
        # calculate bandwidth for Bartlett kernel. if trunclag
        # is not provided, calculate according to SAS 9.4 (2015)
        # methodology.
        if trunclag is None:
            lags = np.amax([1, int(np.sqrt(nobs) / 5)])
        else:
            lags = trunclag
        bw = lags + 1
        # get the residual self variances and covariances
        var, cov = self._sigma_est_pp(ols.resid, nobs, bw)
        s_hat = var + cov
        mse = var * nobs / (nobs - exog.shape[1])
        tau1 = np.sqrt(var / s_hat) * (rho_hat - 1) / se
        tau2 = 0.5 * cov * nobs * se/(np.sqrt(s_hat) * np.sqrt(mse))
        tau = tau1 - tau2
        rho = nobs * (rho_hat - 1) - \
            tau2 * np.sqrt(s_hat) * nobs * se / np.sqrt(mse)
        # get Z-tau p-value and critical values using
        # MacKinnon (1994, 2010) response surface model
        # from statsmodels (2010)
        tau_pv = mackinnonp(tau, regression=regression)
        tau_cvs = mackinnoncrit(regression=regression, nobs=nobs)
        tau_cvdict = {'1%': tau_cvs[0], '5%': tau_cvs[1], '10%': tau_cvs[2]}
        # get Z-rho p-value and critical values using
        # simulation-derived critical values
        rho_crit = self.__pp_crit(rho, regression)
        rho_pv = rho_crit[0]
        rho_cvdict = rho_crit[1]
        return tau, tau_pv, tau_cvdict, rho, rho_pv, rho_cvdict, lags, nobs

    def __call__(self, x, regression='c', trunclag=None):
        return self.run(x, regression=regression, trunclag=trunclag)


# output results
def _print_res(res, st, reg='c'):
    print("  reg = {}  nlags = {}  nobs = {}".format(reg, res[6], res[7]))
    print("  Z-tau = {0:0.5f}".format(res[0]),
          " pval = {0:0.5f}".format(res[1]))
    print("    Z-tau cvdict = \'1%\': {0:0.5f}".format(res[2]["1%"]),
          " \'5%\': {0:0.5f}".format(res[2]["5%"]),
          " \'10%\': {0:0.5f}".format(res[2]["10%"]))
    print("  Z-rho = {0:0.5f}".format(res[3]),
          " pval = {0:0.5f}".format(res[4]))
    print("    Z-rho cvdict = \'1%\': {0:0.5f}".format(res[5]["1%"]),
          " \'5%\': {0:0.5f}".format(res[5]["5%"]),
          " \'10%\': {0:0.5f}".format(res[5]["10%"]))
    print("  time = {0:0.5f}".format(time.time() - st))


# unit tests taken from Schwert (1987) and verified
# against SAS 9.4 and R package urca 1.3-0
def main():
    print("Phillips-Perron unit-root test...")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "results")
    files = ['BAA.csv', 'DBAA.csv', 'SP500.csv', 'DSP500.csv', 'UN.csv',
             'DUN.csv']
    pp_ur = PhillipsPerron()
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'BAA.csv':
            res = pp_ur(mdl, regression='nc')
            _print_res(res=res, st=st, reg='nc')
            assert_almost_equal(res[0], 0.97086, decimal=5)
            assert_almost_equal(res[1], 0.91186, decimal=5)
            assert_almost_equal(res[3], 0.67490, decimal=5)
            assert_almost_equal(res[4], 0.85168, decimal=5)
            assert_equal(res[6], 4)
            st = time.time()
            res = pp_ur(mdl, trunclag=8)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -0.80240, decimal=5)
            assert_almost_equal(res[1], 0.81842, decimal=5)
            assert_almost_equal(res[3], -1.31321, decimal=5)
            assert_almost_equal(res[4], 0.85547, decimal=5)
            assert_equal(res[6], 8)
        elif file == 'DBAA.csv':
            res = pp_ur(mdl)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -11.58349, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -202.25996, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 4)
            st = time.time()
            res = pp_ur(mdl, regression='ct', trunclag=3)
            _print_res(res=res, st=st, reg='ct')
            assert_almost_equal(res[0], -11.75502, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -211.94256, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 3)
        elif file == 'SP500.csv':
            res = pp_ur(mdl, regression='nc')
            _print_res(res=res, st=st, reg='nc')
            assert_almost_equal(res[0], 2.87115, decimal=5)
            assert_almost_equal(res[1], 0.99963, decimal=5)
            assert_almost_equal(res[3], 2.46436, decimal=5)
            assert_almost_equal(res[4], 0.98354, decimal=5)
            assert_equal(res[6], 4)
            st = time.time()
            res = pp_ur(mdl, regression='ct', trunclag=7)
            _print_res(res=res, st=st, reg='ct')
            assert_almost_equal(res[0], -1.16404, decimal=3)
            assert_almost_equal(res[1], 0.91762, decimal=3)
            assert_almost_equal(res[3], -6.90218, decimal=5)
            assert_almost_equal(res[4], 0.67106, decimal=5)
            assert_equal(res[6], 7)
        elif file == 'DSP500.csv':
            res = pp_ur(mdl, regression='nc')
            _print_res(res=res, st=st, reg='nc')
            assert_almost_equal(res[0], -20.11853, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -439.58955, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 4)
            st = time.time()
            res = pp_ur(mdl, regression='ct', trunclag=2)
            _print_res(res=res, st=st, reg='ct')
            assert_almost_equal(res[0], -20.45611, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -441.82363, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 2)
        elif file == 'UN.csv':
            res = pp_ur(mdl, trunclag=8)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.46631, decimal=5)
            assert_almost_equal(res[1], 0.12389, decimal=5)
            assert_almost_equal(res[3], -11.58438, decimal=5)
            assert_almost_equal(res[4], 0.09282, decimal=5)
            assert_equal(res[6], 8)
            st = time.time()
            res = pp_ur(mdl, regression='ct')
            _print_res(res=res, st=st, reg='ct')
            assert_almost_equal(res[0], -2.65137, decimal=5)
            assert_almost_equal(res[1], 0.25682, decimal=5)
            assert_almost_equal(res[3], -14.15295, decimal=5)
            assert_almost_equal(res[4], 0.21564, decimal=5)
            assert_equal(res[6], 4)
        elif file == 'DUN.csv':
            res = pp_ur(mdl, trunclag=6)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -19.96474, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -551.68096, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 6)
            st = time.time()
            res = pp_ur(mdl, regression='ct', trunclag=10)
            _print_res(res=res, st=st, reg='ct')
            assert_almost_equal(res[0], -20.56629, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_almost_equal(res[3], -612.04248, decimal=5)
            assert_almost_equal(res[4], 0.00000, decimal=5)
            assert_equal(res[6], 10)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
