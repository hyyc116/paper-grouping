# paper-grouping
scientific paper grouping algorithm based on citation distribution. Using a adjusted R square to evaluate the most suitable power law fitting line to citation distribution, and group papers into three levels: low-cited paper, medium-cited paper, and highly-cited paper. 

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
The output is an image showing result of citation distribution, procedures of paper grouping(including results of using R square, percent R square, and adjusted R square). The result of adjusted R square is recommended, but for some datasets, the other two may also be valuable. The format could be png, pdf,jpg, etc.

![Grouping result of demo](output/grouping-result.png)


### Citations
If you use the algorithms in this code, please add following reference.
    
    

