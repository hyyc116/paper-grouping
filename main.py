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
import powerlaw
import argparse


mpl.rcParams['agg.path.chunksize'] = 10000
params = {'legend.fontsize': 8,
         'axes.labelsize': 15,
         'axes.titlesize':20,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.DEBUG)

## from a citation distribution dict {count: #(count)}, to split papers to three levels
def grouping_papers(citation_list,distribution_path,medium,step,verbose):
    # 所有文章的被引次数
    citation_dis = Counter(citation_list)
    total = len(citation_list)
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

    ccs = sorted(citation_dis)

    x_min_max = medium-step-5
    x_max_min = medium+step-5

    logging.info('---------- paper grouping ---------------')
    logging.info('number of papers: {:}, the medium is {:}, and step is {:}'.format(len(citation_list),medium,step))
    logging.info('-----------------------------------------')


    for citation_count in ccs:
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
    fig = plt.figure(figsize=(20,25))
    ## first plot citation distribution
    # ax00 = axes[0,0]
    ax00 = fig.add_subplot(5,3,1)
    logging.info('plot the original distribution...')
    plot_citation_distribution(ax00,xs,ys,10,300,_min_y,_max_y)


    ## plot the grid search result of using R2 directly
    ax10 = fig.add_subplot(5,3,4)
    ax11 = fig.add_subplot(5,3,5, projection='3d')
    ax12 = fig.add_subplot(5,3,6)
    xmin,xmax = plot_fitting_and_distribution(fig,ax10,ax11,xs,ys,'r2',_min_y,_max_y,x_min_max,x_max_min,step)
    plot_num_dis(citation_list,xmin,xmax,ax12)

    ##plot percent curves as increase of x_max 
    ax20 = fig.add_subplot(5,3,7)
    logging.info('plotting the percentage rate .. ')
    plot_percentage_curves(ax20,p_xs,p_ys,'$x_{max}$','$P(1,x_{max})$','trend of $P(1,x_{max})$')
    
    ax21 = fig.add_subplot(5,3,8)
    logging.info('plotting the points perentage .. ')
    plot_percentage_curves(ax21,px_xs,px_ys,'$x_{max}$','#($x_i$)/#($X$)','trend of #($x_i$)/#($X$)')

    ## plot the grid search result of using percentage r2
    ax30 = fig.add_subplot(5,3,10)
    ax31 = fig.add_subplot(5,3,11, projection='3d')
    ax32 = fig.add_subplot(5,3,12)
    xmin,xmax = plot_fitting_and_distribution(fig,ax30,ax31,xs,ys,'percent_r2',_min_y,_max_y,x_min_max,x_max_min,step)
    plot_num_dis(citation_list,xmin,xmax,ax32)

    ## plot the grid search result of using percentage r2
    ax40 = fig.add_subplot(5,3,13)
    ax41 = fig.add_subplot(5,3,14, projection='3d')
    ax42 = fig.add_subplot(5,3,15)
    xmin,xmax = plot_fitting_and_distribution(fig,ax40,ax41,xs,ys,'adjusted_r2',_min_y,_max_y,x_min_max,x_max_min,step)
    plot_num_dis(citation_list,xmin,xmax,ax42)

    plt.tight_layout()
    plt.savefig(distribution_path,dpi=200)
    logging.info('distribution saved to {:}.'.format(distribution_path))

def plot_fitting_and_distribution(fig,ax1,ax2,xs,ys,evaluator_name,_min_y,_max_y,x_min_max,x_max_min,step):
    logging.info('Optimize using {:} ... '.format(evaluator_name))
    start,end = fit_xmin_xmax(xs,ys,fig,ax2,evaluator_name,x_min_max,x_max_min,step)
    logging.info('Search result: X_min =  {:},  X_max = {:} ...'.format(start,end))
    popt,pcov = curve_fit(power_low_func,xs[start:end],ys[start:end])
    xmin = xs[start]
    xmax = xs[end-1]
    plot_citation_distribution(ax1,xs,ys,xmin,xmax,_min_y,_max_y,True)
    ax1.plot(np.linspace(xmin, xmax, 10), power_low_func(np.linspace(xmin, xmax, 10), *popt),label='$\\alpha={:.2f}$'.format(popt[0]))
    # ax1.plot([start]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{min}$'+'$={:}$'.format(start))
    # ax1.plot([end]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{max}$'+'$={:}$'.format(end))
    return xmin,xmax


def plot_num_dis(citation_list,xmin,xmax,ax):

    total_all = len(citation_list)

    low_num = 0
    medium_num = 0
    high_num = 0

    for i,c in enumerate(citation_list):

        if c < xmin:
            low_num+=1
        elif c<xmax:
            medium_num+=1
        else:
            high_num+=1


    xs = [low_num,medium_num,high_num]
    labels = ['low','medium','high']

    rects = ax.bar(np.arange(len(xs)),xs,align='center')

    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(labels)
    autolabel(rects,ax,total_all)
    ax.set_xlabel('paper level')
    ax.set_yscale('log')
    ax.set_ylabel('number of papers')



def autolabel(rects,ax,total,step=1):
    """
    Attach a text label above each bar displaying its height
    """
    for index in np.arange(len(rects),step=step):
        rect = rects[index]
        height = rect.get_height()
        # print height
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                '{:}({:.2%})'.format(height,height/float(total)),
                ha='center', va='bottom')



def plot_citation_distribution(ax,xs,ys,xmin,xmax,_min_y,_max_y,isFinal=False):
    ax.plot(xs,ys,'o',fillstyle='none')
    # xmin = xs[start]
    # xmax = xs[end]
    # ax.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--')
    # ax.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--')

    if not isFinal:
        ax.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--')
        ax.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--')
        ax.text(2,10**-3,'II',fontsize=30)
        ax.text(80,10**-2,'I',fontsize=30)
        ax.text(1000,10**-4,'III',fontsize=30)

    else:
        ax.plot([xmin]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{min}$'+'$={:}$'.format(xmin))
        ax.plot([xmax]*10, np.linspace(_min_y, _max_y, 10),'--',label='$x_{max}$'+'$={:}$'.format(xmax))
        ax.text(1,10**-3,'Lowly-cited',fontsize=15)
        ax.text(xmin,10**-1,'Medium-cited',fontsize=15)
        ax.text(xmax,10**-4,'Highly-cited',fontsize=15)
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

def fit_xmin_xmax(xs,ys,fig,ax,evaluator_name,x_min_max,x_max_min,step):

    rxs=[]
    rys=[]
    rzs=[]

    max_y = np.log(np.max(ys))
    min_y = np.log(np.min(ys))
    normed_total_ys = (np.log(ys)-min_y)/(max_y-min_y)

    ### 如何进行grid search的维度定义


    x_is = np.arange(1,x_min_max,step)
    y_is = np.arange(x_max_min,len(xs),step)

    ROWS = len(x_is)
    COLS = len(y_is)

    max_start=0
    max_end =0
    max_z = 0

    for i,start in enumerate(x_is):
        for j,end in enumerate(y_is):

            x = xs[start:end]
            y = ys[start:end]

            popt,pcov = None,None
            try:
                popt,pcov = curve_fit(power_low_func,x,y)
            except:
                logging.warn('grid [{:},{:}],CANNOT FIND THE OPTINAL PARAMETERS. r2 is set to 0.'.format(x[0],x[-1]))
                # continue

            if popt is not None and pcov is not None:
                fit_y = power_low_func(x, *popt)
                r2 = r2_score(np.log(y),np.log(fit_y))

            else:

                r2 = 0

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

    
def main():

    parser = argparse.ArgumentParser(usage='python %(prog)s [options] --input [input citation file path] --output [output figure path]',epilog='take care the medium value and the step of your dataset!')

    parser.add_argument('-i','--input',dest='inputfile',default=None,help='the path of input file, cannot be none.')
    parser.add_argument('-o','--output',dest='output',default=None,help='the path of output figure, cannot be none.')
    parser.add_argument('-s','--step',dest='step',default=5,type=int,help='the step of grid search, default is 5.')
    parser.add_argument('-m','--medium',dest='medium',default=80,type=int,help='the medium value of grid search,integers,default is 80.')
    parser.add_argument('-v','--verbose',dest='verbose',action='store_true',default=True,help='whether print logging info')


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

    step = args.step

    medium = args.medium


    if step < 1:

        logging.info('step should be greater than 0.')

        parser.print_help()

        return


    if medium < 1:

        logging.info('step should be greater than 0.')

        parser.print_help()

        return


    verbose = args.verbose

    citation_list = []
    for line in open(inputfile):
        citations = [int(i) for i in line.split(',')]
        citation_list.extend(citations)


    ### 如何定义中间值是一个问题
    grouping_papers(citation_list,outfile,medium,step,verbose)

if __name__ == '__main__':
    main()
        

