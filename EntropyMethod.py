#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'JiaChengXu'
__mtime__ = '2018/9/2'
"""

import numpy as np

def ApproximateEntropy(X, m, r, tao=1):
	""" Approximate Entropy
		返回信号的近似熵。
	Parameters
	----------
	X :  array_like
		Array containing numbers whose approximate entrpy is desired. X must be a array.
	m :  int
		空间维度m, 其值必须为正整数
	r ： int, float or str
		str, 相似容度的系数，经验上一般取(0.1~0.25)std
		int, float: 相似容度值
	tao : int
		尺度，其值必须为正整数

	Returns
	-------
	ApEn : array_like
		返回每列信号的近似熵

	References
	-------
	[1]. https://en.wikipedia.org/wiki/Approximate_entropy#cite_note-Pincus21991-23

	[2]. Pincus, S. M. (1991). "Approximate entropy as a measure of system complexity".
	Proceedings of the National Academy of Sciences.
	88 (6): 2297–2301. doi:10.1073/pnas.88.6.2297. PMC 51218 Freely accessible. PMID 11607165

	Examples
	-------
	>>> x = np.array([85, 80, 89] *17)
	>>> m, r = 2, '0.25std'
	>>> ApproximateEntropy(x,m,r)
	array([  1.09965411e-05])
	"""

	# Compute Dij
	def maxdist(x_i, x_j):
		return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

	# Compute phi_m_r
	def phi(U, m, r1):
		x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
		Ci = [(len([1 for x_j in x if maxdist(x_i, x_j) <= r1]) ) / (N - m + 1.0) for x_i in x]
		return sum(np.log(Ci))/(N - m + 1.0)

	# 将一维数组转化成二维矩阵形式
	try:
		Num, Len = X.shape
	except:
		X = X.reshape([-1,1])
		Num, Len = X.shape

	# Compute 相似容度
	if type(r) == str:
		r = float(r[:-3])
		stdX = r * np.std(X, axis=0)
	else:
		stdX = [r]*Len

	# Compute scale
	if tao > 1:
		dX = X[range(0, Num, tao), :]
	else:
		dX = X

	# Compute ApproximateEntropy
	N = dX.shape[0]
	ApEn = np.zeros([Len])
	for i in range(Len):
		ApEn[i] = abs(phi(dX[:, i], m + 1, stdX[i]) - phi(dX[:, i], m, stdX[i]))
	return ApEn

def SampleEntropy(X, m, r, tao=1):
	""" Sample Entropy
		返回信号的样本熵。
	Parameters
	----------
	X :  array_like
		Array containing numbers whose sample entrpy is desired. X must be a array.
	m :  int
		空间维度m, 其值必须为正整数
	r ： int, float or str
		str, 相似容度的系数，经验上一般取(0.1~0.25)std
		int, float: 相似容度值
	tao : int
		尺度，其值必须为正整数

	Returns
	-------
	SaEn : array_like
		返回每列信号的样本熵

	References
	-------
	[1]. https://en.wikipedia.org/wiki/Sample_entropy

	[2]. Richman, JS; Moorman, JR (2000).
	"Physiological time-series analysis using approximate entropy and sample entropy".
	American Journal of Physiology. Heart and Circulatory Physiology.
	278 (6): H2039–49. doi:10.1152/ajpheart.2000.278.6.H2039. PMID 10843903

	[3]. Costa, Madalena; Goldberger, Ary; Peng, C.-K. (2005).
	"Multiscale entropy analysis of biological signals".
	Physical Review E. 71 (2). doi:10.1103/PhysRevE.71.021906

	Examples
	-------
	>>> x = np.array([85, 80, 89] *17)
	>>> m, r = 2, '0.25std'
	>>> SampleEntropy(x,m,r)
	array([ 0.0008507])
	"""

	# Compute Dij
	def maxdist(x_i, x_j):
		return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

	# Compute phi_m_r
	def phi(U, m, r1):
		x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
		Bi = [(len([1 for x_j in x if maxdist(x_i, x_j) <= r1]) - 1.0) / (N - m) for x_i in x]
		return sum(Bi)/(N - m + 1.0)

	# 将一维数组转化成二维矩阵形式
	try:
		Num, Len = X.shape
	except:
		X = X.reshape([-1,1])
		Num, Len = X.shape

	# Compute 相似容度
	if type(r) == str:
		r = float(r[:-3])
		stdX = r * np.std(X, axis=0)
	else:
		stdX = [r]*Len

	# Compute scale
	if tao > 1:
		dX = X[range(0, Num, tao), :]
	else:
		dX = X

	# Compute SampleEntropy
	N = dX.shape[0]
	SaEn = np.zeros([Len])
	for i in range(Len):
		SaEn[i] = -np.log(phi(dX[:, i], m + 1, stdX[i]) / phi(dX[:, i], m, stdX[i]))
	return SaEn


def FuzzyEntropy(X, m, r, tao = 1):
	""" Fuzzy Entropy
	    返回信号的模糊熵。
    Parameters
    ----------
    X :  array_like
        Array containing numbers whose fuzzy entrpy is desired. X must be a array.
    m :  int
        空间维度m, 其值必须为正整数
	r ： int, float or str
		str, 相似容度的系数，经验上一般取(0.1~0.25)std
		int, float: 相似容度值
    tao : int
        尺度，其值必须为正整数

    Returns
    -------
    FzEn : array_like
		返回每列信号的模糊熵

	References
	-------
	[1]. Chen W, Zhuang J, Yu W, et al. Measuring complexity using fuzzyen, apen, and sampen[J].
	Medical Engineering and Physics, 2009, 31(1): 61-68.

	Examples
	-------
    >>> x = np.array([85, 80, 89] *17)
    >>> m, r = 2, '0.25std'
    >>> FuzzyEntropy(x,m,r)
	array([ 0.44907402])
	"""

	# Compute Aij
	def Aij(x_i,x_j,r1):
		mdist = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
		return np.e**(-np.log(2)*(mdist/r1)**2)

	# Compute phi_m_r
	def phi(U,m,r1):
		x = [[U[j]-np.mean(U[i:i+m]) for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
		Ci = [ sum([Aij(x_i, x_j, r1) for xj, x_j in enumerate(x) if xj != xi ]) / (N - m)  for xi, x_i in enumerate(x)]
		return sum(Ci) / (N - m + 1.0)

	# 将一维数组转化成二维矩阵形式
	try:
		Num, Len = X.shape
	except:
		X = X.reshape([-1,1])
		Num, Len = X.shape

	# Compute scale
	if type(r) == str:
		r = float(r[:-3])
		stdX = r * np.std(X, axis=0)
	else:
		stdX = [r]*Len

	# 计算尺度
	if tao > 1:
		dX = X[range(0, Num, tao), :]
	else:
		dX = X

	# Compute FuzzyEntropy
	N = dX.shape[0]
	FzEn = np.zeros([Len])
	for i in range(Len):
		FzEn[i] = -np.log(phi(dX[:, i], m + 1, stdX[i]) / phi(dX[:, i], m, stdX[i]))
	return FzEn

def DistributionEntropy(X, m, M='auto', tao=1):
	""" DistributionEntropy Entropy
		返回信号的分布熵。
	Parameters
	----------
	X :  array_like
		Array containing numbers whose distribution entrpy is desired. X must be a array.

	m :  int
		空间维度m, 其值必须为正整数

	M ：int or str, optional
		估算ePDF时直方图的个数

        If `bins` is an int, it defines the number of equal-width
        bins in the given range

        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see `Notes` for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that that takes into account data
            variability and data size.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

	tao : int
		尺度，其值必须为正整数

	Returns
	-------
	FzEn : array_like
		返回每列信号的分布熵

	References
	-------
	[1]. Udhayakumar R K, Karmakar C, Li P, et al.
	Effect of data length and bin numbers on distribution entropy (DistEn) measurement in analyzing healthy aging[J].
	Conf Proc IEEE Eng Med Biol Soc, 2015:7877-7880.

	[2]. Peng L, Liu C, Ke L, et al.
	Assessing the complexity of short-term heartbeat interval series by distribution entropy[J].
	Medical & Biological Engineering & Computing, 2015, 53(1):77-87.

	Examples
	-------
	>>> x = np.array([85, 80, 89] *17)
	>>> m = 2
	>>> DistributionEntropy(x,m)
	array([ 0.41186842])
	"""


	# Compute distance
	def maxdist(x_i, x_j):
		return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

	# Compute the empirical probability distribution function
	def ePDF(U, m):
		x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
		dij = np.array([[maxdist(x_i, x_j) for xj, x_j in enumerate(x) if xj != xi] for xi, x_i in enumerate(x)])
		hist_count, _ = np.histogram(dij, bins=M ,density=False)
		return hist_count/(N - m)/(N - m +1)

	# Compute Shanon Entropy
	def ShanonEntropy(p):
		p[p<=1e-4]=1
		return -np.sum(p * np.log(p)) / np.log(len(p))

	# 将一维数组转化成二维矩阵形式
	try:
		Num, Len = X.shape
	except:
		X = X.reshape([-1,1])
		Num, Len = X.shape

	# 计算尺度
	if tao > 1:
		dX = X[range(0, Num, tao), :]
	else:
		dX = X

	# Compute DistributionEntropy
	N = dX.shape[0]
	DisEn = np.zeros([Len])
	for i in range(Len):
		DisEn[i] = ShanonEntropy(ePDF(dX[:, i], m))
	return DisEn


if __name__ == '__main__':
	import time
	np.random.seed(0)
	x = np.random.random([100,6])
	m, r = 2, '0.25std'
	t0 = time.time()
	fzen = FuzzyEntropy(x,m,r)
	t1 = time.time()
	saen = SampleEntropy(x,m,r)
	t2 = time.time()
	apen = ApproximateEntropy(x,m,r)
	t3 = time.time()
	DisEn = DistributionEntropy(x,m)
	t4 = time.time()
	print( 1000*np.array([t1-t0,t2-t1,t3-t2,t4-t3]))
