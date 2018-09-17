#coding:utf-8

'''
使用powlaw包进行power law的拟合

Fit power law distribution with powlaw package.

'''

import powerlaw
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import sys
reload(sys)

sys.setdefaultencoding('utf-8')

## 输入是一个数据的list
def basic_fit(data,xmin=None,xmax=-1):

	if xmax==-1:
		fit = powerlaw.Fit(data,xmin=xmin)
	else:
		fit = powerlaw.Fit(data,xmin=xmin,xmax=xmax)

	# print fit.xmin

	# print fit.power_law.D

	# print fit.power_law.alpha

	# print fit.power_law.sigma

	return fit


def estimate_xmax(data):

	## retrieve xmin for first round
	fit = basic_fit(data,xmin=None,xmax=-1)

	xmin = fit.xmin

	print 'x_min:',xmin



	ends = []
	Ds = []

	nums = sorted(Counter(data).keys())
	for i in np.arange(0,len(nums),5):
		end = nums[i]
		if end < xmin+5:
			continue

		print end

		ends.append(end)
		fit = basic_fit(data,xmin,xmax=end)

		Ds.append(fit.power_law.D)


	min_d = 1
	pos = -1
	for i,d in enumerate(Ds):

		if d<min_d:
			min_d=d
			pos = i

	print ends[pos],min_d




	plt.figure()

	plt.plot(ends,Ds)

	plt.xscale('log')
	plt.yscale('log')

	plt.xlabel('xmax')
	plt.ylabel('Kolmogorov-Smirnov distance')

	plt.tight_layout()

	plt.savefig('output/xmax_estimate_fig.png',dpi=200)







	# print fit.distribution_compare('power_law', 'exponential')
	# print fit.distribution_compare('power_law', 'lognormal')



	# plt.figure()
	# fig = powerlaw.plot_pdf(data,linear_bins = True, color = 'r')
	# powerlaw.plot_pdf(data, linear_bins = False, color = 'b')
	# fit.power_law.plot_pdf(color = 'g',ax=fig)
	# # fit.lognormal.plot_ccdf(color='b', ax=fig)
	# # fit.exponential.plot_ccdf(color='k',ax=fig)
	# plt.tight_layout()
	# plt.savefig('output/test_plot_fig.png',dpi=200)


# def plot_distribution(data,alpha,sigma):
	# pass

def plot_powlaw(data):
	pass



if __name__ == '__main__':
	data = [int(cc) for line in open('data/citation_list.txt') for cc in line.strip().split(',') if int(cc)>0]

	# data = [int(cc.strip()) for cc in open('data/aps-citations.txt') if int(cc.strip())>0]


	data = np.array(data)

	print 'length of data:',len(data)

	# basic_fit(data)

	# plot_powlaw(data_list)

	estimate_xmax(data)


