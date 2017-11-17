# paper-grouping
Scientific paper grouping algorithm is based on the citation distribution. We use an adjusted R square to evaluate the most suitable power law fitting line to the citation distribution, and group papers into three levels: low-cited papers, medium-cited papers, and highly-cited papers. 

### Environment requirement

    * numpy
    * scipy
    * matplotlib
    * scikit-learn


### Usage

    python main.py data/citation_list.txt output/grouping_result.png

### Format of citation_list.txt
    
    1,2,1,1,2,3,4,55,6,7,7,8 ...

Lines in citation_list.txt are citation count of each paper in a dataset separated by comma.

### Result format
The output is an image showing results of the citation distribution, procedures of paper grouping (including results of using R square, percent R square, and adjusted R square). The result of adjusted R square is recommended to adopt, but for some datasets, the other two R squares may also be useful. The format could be png, pdf, jpg, etc.

![Grouping results of the demo](output/grouping-result.png)


### Citations
If you use the algorithms in this repo, please cite us as:
    to be added
    
