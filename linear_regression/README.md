# paper-grouping (Desperated)
Scientific paper grouping algorithm is based on the citation distribution. We use an adjusted R square to evaluate the most suitable power law fitting line to the citation distribution, and group papers into three levels: low-cited papers, medium-cited papers, and highly-cited papers.

### Environment requirement

    * numpy
    * scipy
    * matplotlib
    * scikit-learn


### Usage

    usage: python main.py [options] --input [input citation file path] --output [output figure path]

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUTFILE, --input INPUTFILE
                            the path of input file, cannot be none.
      -o OUTPUT, --output OUTPUT
                            the path of output figure, cannot be none.
      -s STEP, --step STEP  the step of grid search, default is 5.
      -m MEDIUM, --medium MEDIUM
                            the medium value of grid search,integers,default is
                            80.
      -v, --verbose         whether print logging info

    take care the medium value and the step of your dataset!

The medium value and the step is used to set the grid search. The range of xmin is set to [medium-step-5], and the range of xmax is set to [medium+step+5]. The step is size of grid search, the minimum is 1. Take care of these two parameters!


### Example

    python main.py -i data/citation_list.txt -o output/grouping-result.png

    python main.py -i data/aps-citations.txt -o output/aps-grouping-result.png -s 3 -m 80

### Format of citation_list.txt

    1,2,1,1,2,3,4,55,6,7,7,8 ...

Lines in citation_list.txt are citation count of each paper in a dataset separated by comma.

### Result format
The output is an image showing results of the citation distribution, procedures of paper grouping (including results of using R square, percent R square, and adjusted R square). The result of adjusted R square is recommended to adopt, but for some datasets, the other two R squares may also be useful. The format could be png, pdf, jpg, etc.

![Grouping results of the demo](output/aps-grouping-result.png)


### Dataset
Three datasets are utilized in this paper: ArnetMiner and Microsoft Academic Graph (MAG) for the field of computer science, as well as American Physical Society (APS) for the field of physics. ArnetMiner covers important conferences and journals from the domain of computer science. The citation network of ArnetMiner comprises 1,286,254 papers/references ranging from 1936 to 2014 and 8,024,869 local citation relationships. Among all of the papers, 940,198 papers have received at least one citation. Regarding the MAG dataset, only papers in the computer science field are adopted in this paper (MAG-CS); the number of papers in MAG-CS is 5,249,815. We build a citation network composed of 9,703,104 papers and 44,944,243 citation relations. Out of all of these papers, 2,584,967 have been cited at least once. As for the APS dataset, it covers the bibliographic metadata of 13 top journals in physics, including 596,786 publications and 7,211,978 citation relationships among them, in which there are 515,867 publications that have received at least one citation.

### Citations
If you use the algorithms in this repo, please cite us as:
    to be added

