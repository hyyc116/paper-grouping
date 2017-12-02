#coding:utf-8
'''
@author: huangyong
algorithms for paper grouping

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

mpl.rcParams['agg.path.chunksize'] = 10000


color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.DEBUG)

## from a citation distribution dict {count: #(count)}, to split papers to three levels
def grouping_papers(citation_list,distribution_path,x_min_max=80,x_max_min=100):
    # 所有文章的被引次数
    citation_dis = Counter(citation_list)
    total = np.sum(citation_dis.values())
    xs = []
    ys = []
    _max_y = 0
    _min_y = 1

    p_xs = []
    p_ys = []
    p_sum = 0

    px_xs = []
    px_ys = []
    px_sum = 0
    for citation_count in sorted(citation_dis):
        if citation_count==0:
            continue

        xs.append(citation_count)
        y = citation_dis[citation_count]/float(total)
        ys.append(y)
        if y>_max_y:
            _max_y = y

        if y<_min_y:
            _min_y = y

        p_xs.append(citation_count)
        p_sum+=y
        p_ys.append(p_sum)

        px_xs.append(citation_count)
        px_sum+=1
        px_ys.append(px_sum)

    px_ys = np.array(px_ys)/float(px_sum)

    # fig,axes = plt.subplots(4,2,figsize=(14,20))
    fig = plt.figure(figsize=(14,25))
    ## first plot citation distribution
    # ax00 = axes[0,0]
    ax00 = fig.add_subplot(5,2,1)
    logging.info('plot the original distribution...')
    plot_citation_distribution(ax00,xs,ys,10,300,_min_y,_max_y)


    ## plot the grid search result of using R2 directly
    ax10 = fig.add_subplot(5,2,3)
    ax11 = fig.add_subplot(5,2,4, projection='3d')
    plot_fitting_and_distribution(fig,ax10,ax11,xs,ys,'r2',_min_y,_max_y,x_min_max,x_max_min)

    ##plot percent curves as increase of x_max 
    ax20 = fig.add_subplot(5,2,5)
    logging.info('plotting the percentage rate .. ')
    plot_percentage_curves(ax20,p_xs,p_ys,'$x_{max}$','$P(1,x_{max})$','trend of $P(1,x_{max})$')
    
    ax21 = fig.add_subplot(5,2,6)
    logging.info('plotting the points perentage .. ')
    plot_percentage_curves(ax21,px_xs,px_ys,'$x_{max}$','#($x_i$)/#($X$)','trend of #($x_i$)/#($X$)')

    ## plot the grid search result of using percentage r2
    ax30 = fig.add_subplot(5,2,7)
    ax31 = fig.add_subplot(5,2,8, projection='3d')
    plot_fitting_and_distribution(fig,ax30,ax31,xs,ys,'percent_r2',_min_y,_max_y,x_min_max,x_max_min)

    ## plot the grid search result of using percentage r2
    ax40 = fig.add_subplot(5,2,9)
    ax41 = fig.add_subplot(5,2,10, projection='3d')
    plot_fitting_and_distribution(fig,ax40,ax41,xs,ys,'adjusted_r2',_min_y,_max_y,x_min_max,x_max_min)

    plt.tight_layout()
    plt.savefig(distribution_path,dpi=200)
    logging.info('distribution saved to {:}.'.format(distribution_path))

def plot_fitting_and_distribution(fig,ax1,ax2,xs,ys,evaluator_name,_min_y,_max_y,x_min_max=80,x_max_min=100):
    logging.info('Optimize using {:} ... '.format(evaluator_name))
    start,end = fit_xmin_xmax(xs,ys,fig,ax2,evaluator_name,x_min_max,x_max_min)
    logging.info('Search result: X_min =  {:},  X_max = {:} ...'.format(start,end))
    popt,pcov = curve_fit(power_low_func,xs[start:end],ys[start:end])
    plot_citation_distribution(ax1,xs,ys,start,end,_min_y,_max_y)
    ax1.plot(np.linspace(start, end, 10), power_low_func(np.linspace(start, end, 10), *popt),label='$\\alpha={:.2f}$'.format(popt[0]))
    # ax1.plot([start]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{min}$'+'$={:}$'.format(start))
    # ax1.plot([end]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{max}$'+'$={:}$'.format(end))

def plot_citation_distribution(ax,xs,ys,xmin,xmax,_min_y,_max_y,isFinal=False):
    ax.plot(xs,ys,'o',fillstyle='none')

    # ax.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--')
    # ax.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--')



    if not isFinal:
        ax1.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--')
        ax1.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--')
        ax.text('II',xmin/2,10**-4)
        ax.text('I',(xmin+xmax)/2,10**-4)
        ax.text('III',1000,10**-2)

    else:
        ax1.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{min}$'+'$={:}$'.format(xmin))
        ax1.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{max}$'+'$={:}$'.format(xmax))
        ax.text('Low cited',xmin/2,10**-4)
        ax.text('Medium cited',(xmin+xmax)/2,10**-4)
        ax.text('High cited',1000,10**-2)
        ax.legend()

    ax.set_title('Citation Distribution')
    ax.set_xlabel('Citation Count')
    ax.set_ylabel('Relative Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')

def plot_percentage_curves(ax,xs,ys,xlabel,ylabel,title):
    ax.plot(xs,ys)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')

def fit_xmin_xmax(xs,ys,fig,ax,evaluator_name='adjusted_r2',x_min_max=80,x_max_min=100):

    rxs=[]
    rys=[]
    rzs=[]

    max_y = np.log(np.max(ys))
    min_y = np.log(np.min(ys))
    normed_total_ys = (np.log(ys)-min_y)/(max_y-min_y)

    x_is = np.arange(1,x_min_max,2)
    y_is = np.arange(x_max_min,len(xs),5)

    ROWS = len(x_is)
    COLS = len(y_is)

    max_start=0
    max_end =0
    max_z = 0

    for i,start in enumerate(x_is):
        for j,end in enumerate(y_is):

            x = xs[start:end]
            y = ys[start:end]

            popt,pcov = curve_fit(power_low_func,x,y)
            fit_y = power_low_func(x, *popt)
            r2 = r2_score(np.log(y),np.log(fit_y))

            normed_y = (np.log(y)-min_y)/(max_y-min_y)
            percent_of_num = np.sum(normed_y)/float(np.sum(normed_total_ys))
            percentage_r2 = r2*percent_of_num

            percent_of_x = float(len(y))/float(len(ys))
            efficiency = percent_of_num/percent_of_x
            adjusted_r2 = percentage_r2*efficiency

            if evaluator_name=='adjusted_r2':
                evaluator = adjusted_r2
            elif evaluator_name =='percent_r2':
                evaluator = percentage_r2
            elif evaluator_name == 'r2':
                evaluator = r2

            rxs.append(x[0])
            rys.append(x[-1])
            rzs.append(evaluator)

            if evaluator>max_z:
                max_start = x[0],start
                max_end = x[-1],end
                max_z = evaluator

    ax.view_init(60, 210)
    X = np.reshape(rys,(ROWS,COLS))
    Y = np.reshape(rxs,(ROWS,COLS))
    Z = np.reshape(rzs,(ROWS,COLS))
    ax.set_xlabel('$x_{max}$')
    ax.set_ylabel('$x_{min}$')
    ax.set_zlabel(evaluator_name)
    logging.info('max_start: {:}, max_end:{:}.'.format(max_start,max_end))
    ax.set_title('$x_{min}$:'+'{:}'.format(max_start[0])+' - $x_{max}$:'+'{:}'.format(max_end[0])+', {:}={:.4f}'.format(evaluator_name,max_z))
    surf = ax.plot_surface(X,Y,Z, cmap=CM.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=10,ax=ax)
    return max_start[-1],max_end[-1]

def power_low_func(x,a,b):
    return b*(x**(-a))
    
def main(citation_dis_path,output_path):
    citation_list = []
    for line in open(citation_dis_path):
        citations = [int(i) for i in line.split(',')]
        citation_list.extend(citations)
    grouping_papers(citation_list,output_path)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        logging.info('Two parameters should be provided: 1.citation list of your dataset, 2.the output result of grouping papers.')
        logging.info('e.g. {:} a.txt output.png'.format(sys.argv[0]))
    else:
        main(sys.argv[1],sys.argv[2])
        

