import numpy as np


# binomial expansion for ARFIMA models
def _calc_arfima_binomial(n, nterms):
    # switch equation side
    n = -n
    bc = np.zeros([nterms, 1])
    bc[0] = 1
    # generate coefficients
    for i in range(1, nterms):
        bc[i] = abs(bc[i - 1] * (n - (i - 1)) / i)
    return bc


def ARFIMA_sim(p_coeffs, q_coeffs, d, slen, alpha=0, sigma=1, numseas=100):
    """
    Generate a random ARFIMA(p,d,q) series. Generalizes to ARMA(p,q)
    when d = 0, and ARIMA(p,d,q) when d = 1.

    User provides an array of coefficients for the AR(p) and MA(q)
    portions of the series as well as the fractional differencing
    parameter and the required length. A constant may optionally be
    specified, as well as the standard deviation of the Gaussian
    innovations, and the number of seasoning samples to be
    generated before recording the series.

    Parameters
    ----------
    p_coeffs : array_like
        AR(p) coefficients
        len(p_coeffs) <= 10
    q_coeffs : array_like
        MA(q) coefficients
        len(q_coeffs) <= 10
    d : float
        fractional differencing parameter
        -1 < d <= 1
    slen : int
        number of samples in output ARFIMA series
        10 <= len(series) <= 100000
    alpha : float
        series constant (default=0)
    sigma : float
        standard deviation of innovations
    numseas : int
        number of seasoning samples (default=100)
        0 <= num(seasoning) <= 10000

    Returns
    -------
    series : 1d array
        random ARFIMA(p,d,q) series of specified length

    Notes
    -----
    MA(q) parameters follow the Box-Jenkins convention which uses a
    difference representation for the MA(q) process which is the opposite
    of the standard ARIMA MA(q) summation representation. This matches the
    operation of SAS/farmasim and R/arfimasim. As such, the SAS/farmafit
    and R/arfima MA(q) estimates match the sign of the specified MA(q)
    parameters while the statsmodels ARIMA().fit() estimates have opposite
    the specified MA(q) parameter signs.

    References
    ----------
    SAS Institute Inc (2013). SAS/IML User's Guide. Cary, NC: SAS Institute
    Inc.

    Veenstra, J.Q. (2012). Persistence and Anti-persistence: Theory and
    Software (Doctoral Dissertation). Western University, Ontario, Canada.
    """
    p = np.asarray(p_coeffs)
    if p.ndim > 2 or (p.ndim == 2 and p.shape[1] != 1):
        raise ValueError(
            'ARFIMA_sim: p must be 1d array or 2d array with single column')
    p = np.reshape(p, (-1, 1))
    if p.shape[0] > 10:
        raise ValueError(
            'ARFIMA_sim: AR order must be <= 10')
    q = np.asarray(q_coeffs)
    if q.ndim > 2 or (q.ndim == 2 and q.shape[1] != 1):
        raise ValueError(
            'ARFIMA_sim: q must be 1d array or 2d array with single column')
    q = np.reshape(q, (-1, 1))
    if q.shape[0] > 10:
        raise ValueError(
            'ARFIMA_sim: MA order must be <= 10')
    if d <= -1 or d > 1:
        raise ValueError(
            'ARFIMA_sim: valid differencing parameter in range (-1, 1]')
    if slen < 10 or slen > 100000:
        raise ValueError(
            'ARFIMA_sim: valid series length in range [10, 100000]')
    if numseas < 0 or numseas > 10000:
        raise ValueError(
            'ARFIMA_sim: valid seasoning length in range [0, 10000]')
    # check for negative fractional d. if negative,
    # add a unity order of integration, then single
    # difference the final series.
    neg = 0
    if d < 0:
        d += 1
        neg = 1
    # generate the MA(q) series
    lqc = q.shape[0]
    if lqc == 0:
        ma = np.random.normal(scale=sigma, size=slen+numseas)
    else:
        e = np.random.normal(scale=sigma, size=slen+numseas)
        ma = np.zeros([slen+numseas, 1])
        ma[0] = e[0]
        for t in range(1, slen + numseas):
            err = e[max(0, t-lqc):t]
            qcr = np.flip(q[0:min(lqc, t)])
            ma[t] = e[t] - np.dot(err, qcr)
    # generate the ARMA(p,q) series
    lpc = p.shape[0]
    if lpc == 0:
        arma = ma
    else:
        arma = np.zeros([slen+numseas, 1])
        arma[0] = ma[0]
        for t in range(1, slen + numseas):
            arr = arma[max(0, t-lpc):t]
            pcr = np.flip(p[0:min(lpc, t)])
            arma[t] = ma[t] + np.dot(arr.T, pcr)
    # generate the ARFIMA(p,d,q) series
    if np.isclose(d, 0):
        series = alpha + arma
    else:
        # get binomial coefficients
        bc = np.flip(_calc_arfima_binomial(d, slen + numseas))
        end = slen + numseas + 1
        series = np.zeros([slen+numseas, 1])
        for t in range(slen + numseas):
            bcr = bc[end-t-2:end]
            ars = arma[0:t+1]
            series[t] = alpha + np.dot(bcr.T, ars)
        # if negative d then single difference
        if neg:
            series1 = np.zeros([slen+numseas, 1])
            series1[0] = series[0]
            for t in range(1, slen + numseas):
                series1[t] = series[t] - series[t - 1]
            series = series1
    # trim seasoning samples and return 1d
    return series[numseas:].flatten()
