# Phillips-Perron
Python implementation of the Phillips-Perron (1988) test that can be used to
test for a unit root in a univariate process.

Also included here is the Monte Carlo simulation code used to generate
critical values for the Z-rho statistic which makes use of the simulation code
located in the repo: https://github.com/JimVaranelli/ARFIMA_sim

## Parameters
x : array_like, 1d \
&nbsp;&nbsp;&nbsp;&nbsp;data series \
regression : {'nc', 'c','ct'} \
&nbsp;&nbsp;&nbsp;&nbsp;Constant and trend order to include in regression \
&nbsp;&nbsp;&nbsp;&nbsp;* 'nc' : no constant, no trend \
&nbsp;&nbsp;&nbsp;&nbsp;* 'c'  : constant only (default) \
&nbsp;&nbsp;&nbsp;&nbsp;* 'ct' : constant and trend \
trunclag : {int, None} \
&nbsp;&nbsp;&nbsp;&nbsp;number of truncation lags, default=int(sqrt(nobs)/5)
(SAS, 2015)

## Returns
tau : float \
&nbsp;&nbsp;&nbsp;&nbsp;Z-tau test statistic \
tau_pv : float \
&nbsp;&nbsp;&nbsp;&nbsp;Z-tau p-value based on MacKinnon (1994, 2010)
regression surface model \
tau_cvdict : dict \
&nbsp;&nbsp;&nbsp;&nbsp;critical values for the Z-tau test statistic at the 1%,
5%, and 10% levels \
rho : float \
&nbsp;&nbsp;&nbsp;&nbsp;Z-rho test statistic \
rho_pv : float \
&nbsp;&nbsp;&nbsp;&nbsp;Z-tau p-value based on interpolation of
simulation-derived critical values \
rho_cvdict : dict \
&nbsp;&nbsp;&nbsp;&nbsp;critical values for the Z-rho test statistic at the 1%,
5%, and 10% levels \
lags : int \
&nbsp;&nbsp;&nbsp;&nbsp;number of truncation lags used in covariance matrix
estimation \
nobs : int \
&nbsp;&nbsp;&nbsp;&nbsp; number of observations used in regression


## Notes
H0 = series has a unit root (i.e., non-stationary)

Basic process is to fit the time series under test with an AR(1) model
using heteroscedasticity- and autocorrelation-consistent residual
covariance estimation in order to generate the Phillips-Perron Z-rho
and Z-tau statistics (1988) which are asymptotically equivalent to the
Dickey-Fuller (1979, 1981) rho/tau statistics. Z-tau p-values are
calculated using the statsmodel implementation of MacKinnon's (1994,
2010) regression surface model. Z-rho p-values are interpolated from
Monte-Carlo derived critical values. The simulation code used to estimate
the Z-rho critical values is provided here.

## References
Dickey, D.A., and Fuller, W.A. (1979). Distribution of the estimators for
autoregressive time series with a unit root. Journal of the American
Statistical Association, 74: 427-431.

Dickey, D.A., and Fuller, W.A. (1981). Likelihood ratio statistics for
autoregressive time series with a unit root. Econometrica, 49: 1057-1072.

MacKinnon, J.G. (1994). Approximate asymptotic distribution functions for
unit-root and cointegration tests. Journal of Business and Economic
Statistics, 12: 167-176.

MacKinnon, J.G. (2010). Critical values for cointegration tests. Working
Paper 1227, Queen's University, Department of Economics. Retrieved from
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
(Eds.), Proceedings of the 9th Python in Science Conference (pp. 57-61).

## Requirements
Python 3.7 \
Numpy 1.18.1 \
Statsmodels 0.11.0 \
Pandas 1.0.1

## Running
There are no parameters. The program is set up to access test files in the
.\results directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced
output for the test files.
