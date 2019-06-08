#coding:utf-8
### grouping wos papers
from main import *
import json

## 将wos的数据进行分类
def group_wos_papers():

    subj_cns = json.loads(open('data/wos_subj_cd.json').read())

    ## 画出所有领域的论文的分布
    all_cns = subj_cns['ALL']
    grouping_papers(all_cns,'output/wos_all_result.png')
    logging.info('wos all papers result saved to output/wos_all_result.png')

    ## 六个子领域的分布
    fig,axes = plt.subplots(2,3,figsize=(12,3.5))
    for i,subj in enumerate([subj for subj in sorted(subj_cns.keys()) if subj !='ALL']):

        ax = axes[int(i/3),i%3]
        cns = subj_cns[subj]
        xmin,xmax,xs,ys,slope_xs,delta_avg = find_xmin_xmax(cns)
        plot_citation_distribution(ax0,xs,ys,xmin,xmax,subj)

    plt.tight_layout()
    plt.savefig('output/wos_six_subj_result.png',dpi=400)
    logging.info('wos six subject papers result saved to output/wos_six_subj_result.png')


if __name__ == '__main__':
    group_wos_papers()