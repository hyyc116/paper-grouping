#coding:utf-8
'''
@author: huangyong
algorithms for paper grouping

heuristic method for finding xmin and xmax.

'''
import os
import sys
import json
from collections import defaultdict
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import math
import numpy as np
import random
import logging
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from collections import Counter
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
import matplotlib.colors as colors
# import powerlaw
import argparse


mpl.rcParams['agg.path.chunksize'] = 10000
params = {'legend.fontsize': 6,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

## from a citation distribution dict {count: #(count)}, to split papers to three levels
def find_xmin_xmax(citation_list,x_min_max = 200):
    logging.info('---------- paper grouping ---------------')
    # 所有文章的被引次数
    citation_dis = Counter(citation_list)
    xs = []
    ys = []
    xmax = 0
    ccs = sorted(citation_dis)
    for citation_count in ccs:
        if citation_count==0:
            continue
        xs.append(citation_count)
        y = citation_dis[citation_count]
        ys.append(y)

        ## 第一个出现一次的citation count是xmax
        if y ==1 and xmax==0:
            xmax = citation_count

    ## 根据ccdf来确定xmin
    ccdf_ys = []
    for  i,y in enumerate(ys):
        ccdf_ys.append(np.sum(ys[i+1:])/float(np.sum(ys)))

    ##计算相邻两点之间连线的斜率
    xmins = []
    slopes = []
    for i,x in enumerate(xs):
        if x >x_min_max:
            continue

        if i ==0:
            continue

        ## slope的计算是与前一个点连线的斜率与到第一个点连线的斜率的比值
        slope = abs((np.log(ccdf_ys[i-1])-np.log(ccdf_ys[i]))/float(np.log(xs[i])-np.log(xs[i-1])))/abs((np.log(ccdf_ys[0])-np.log(ccdf_ys[i]))/float(np.log(xs[i])-np.log(xs[0])))

        if slope is not None:
            xmins.append(xs[i])
            slopes.append(1/slope)

    slope_xs = []
    delta_slopes = []
    for i,slope in enumerate(slopes):

        if i==0 or i ==len(slopes)-1:
            continue

        slope_xs.append(xmins[i])
        delta_slopes.append(abs(slopes[i]/slopes[i-1]))

    delta_avg = [np.mean(delta_slopes[0 if i-5 < 0 else i-3 :i+1]) for i in range(len(delta_slopes))]
    mean,std,mean_std = np.mean(delta_avg),np.std(delta_avg),np.mean(delta_avg)-np.std(delta_avg)

    ## 在一个std的范围内的置信区间是99%
    xmin = 0
    for i,delta in enumerate(delta_avg):
        if delta > mean_std:
            xmin = slope_xs[i]
            break

    logging.info('xmin is %s , xmax is %s .' % (xmin,xmax))

    return xmin,xmax,xs,ys,slope_xs,delta_avg


def num_percent(xs,ys,xmin,xmax):

    low_cited_paper_num = 0
    medium_cited_paper_num = 0
    high_cited_paper_num = 0
    for i,x in enumerate(xs):

        y = ys[i]

        if x<xmin:

            low_cited_paper_num+=y

        elif x<xmax:

            medium_cited_paper_num+=y
        else:

            high_cited_paper_num+=y

    low_rate,m_rate,h_rate = low_cited_paper_num/float(np.sum(ys)),medium_cited_paper_num/float(np.sum(ys)),high_cited_paper_num/float(np.sum(ys))

    logging.info('lowly cited paper rate: %.4f, medium cited paper rate: %.4f, high cited paper rate: %.4f .'%(low_rate,m_rate,h_rate))
    return low_rate,m_rate,h_rate


def plot_figure_one(xs,ys,xmin,xmax,slope_xs,delta_avg,output):

    ##画图
    fig,axes = plt.subplots(1,2,figsize=(8,3.5))
    ax0 = axes[0]
    plot_citation_distribution(ax0,xs,ys,xmin,xmax)
    ax1 = axes[1]
    ax1.plot(slope_xs,delta_avg)
    ax1.plot(slope_xs,[np.mean(delta_avg)-np.std(delta_avg)]*len(slope_xs),'--',c='r')
    ax1.set_xscale('log')
    ax1.set_xlabel('number of citations')
    ax1.set_ylabel('slope change rate')
    ax1.set_title('Slope Change Rate')
    plt.tight_layout()
    plt.savefig(output,dpi=400)
    logging.info('result figure saved to %s .' % output)

## 目前与两种方法进行对比，第一种是平均分，第二种是1%，10%，other
def compare_methods(citation_list):
    sorted_citations = sorted(citation_list,reverse=True)

    num = len(citation_list)
    ## 第一种33%
    xmax_1 = sorted_citations[int(num/3)]
    xmin_1 = sorted_citations[int(num*2/3)]

    logging.info('xmin and xmax for 33% spliting is {:},{:}'.format(xmin_1,xmax_1))

    ## 第二种是%1，%10，other
    xmax_2 = sorted_citations[int(num*0.01)]
    xmin_2 = sorted_citations[int(num*0.1)]

    logging.info('xmin and xmax for 1%, 10% and other spliting is {:},{:}'.format(xmin_2,xmax_2))



## 文章分类
def grouping_papers(citation_list,distribution_path,x_min_max = 200):
    xmin,xmax,xs,ys,slope_xs,delta_avg = find_xmin_xmax(citation_list,x_min_max)
    num_percent(xs,ys,xmin,xmax)
    plot_figure_one(xs,ys,xmin,xmax,slope_xs,delta_avg,distribution_path)

def plot_citation_distribution(ax,xs,ys,xmin,xmax,title='Citation Distribution'):
    ax.plot(xs,ys,'o',fillstyle='none',alpha = 0.8)
    ax.plot([xmin]*10, np.linspace(np.min(ys), np.max(ys), 10),'--',label='$x_{min}$'+'$={:}$'.format(xmin))
    ax.plot([xmax]*10, np.linspace(np.min(ys), np.max(ys), 10),'-.',label='$x_{max}$'+'$={:}$'.format(xmax))
    ax.text(1,200,'Lowly cited',fontsize=8)
    ax.text(xmin*2,200,'Medium cited',fontsize=8)
    ax.text(xmax*2,200,'Highly cited',fontsize=8)
    ax.legend()

    ax.set_title(title)
    ax.set_xlabel('number of citations')
    ax.set_ylabel('number of publications')
    ax.set_xscale('log')
    ax.set_yscale('log')


def main():

    parser = argparse.ArgumentParser(usage='python %(prog)s [options] --input [input citation file path] --output [output figure path]')

    parser.add_argument('-i','--input',dest='inputfile',default=None,help='the path of input file, cannot be none.')
    parser.add_argument('-o','--output',dest='output',default=None,help='the path of output figure, cannot be none.')
    parser.add_argument('-x','--xminmax',dest='xminmax',default=200,type = int,help='the maximum of xmin.')
    parser.add_argument('-C','--compare',dest='compare',default=True,action='store_true',help='compare with other two methods.')

    args = parser.parse_args()

    inputfile = args.inputfile

    if inputfile is None:

        logging.info('input file path cannot be null.')

        parser.print_help()

        return

    outfile = args.output

    if outfile is None:

        logging.info('output figure path cannot be null.')

        parser.print_help()

        return

    x_min_max = args.xminmax

    # verbose = args.verbose
    citation_list = []
    for line in open(inputfile):
        citations = [int(i) for i in line.split(',')]
        citation_list.extend(citations)

    ### 如何定义中间值是一个问题
    grouping_papers(citation_list,outfile,x_min_max)

    if args.compare:
        compare_methods(citation_list)

if __name__ == '__main__':
    main()


