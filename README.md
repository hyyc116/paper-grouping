# paper-grouping
Scientific paper grouping algorithm is based on the citation distribution. We use an heuristic method to find xmin and xmax, and group papers into three levels: low-cited papers, medium-cited papers, and highly-cited papers.

### Environment requirement

    * numpy
    * scipy
    * matplotlib


### Usage

	usage: python main.py [options] --input [input citation file path] --output [output figure path]

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUTFILE, --input INPUTFILE
                            the path of input file, cannot be none.
      -o OUTPUT, --output OUTPUT
                            the path of output figure, cannot be none.
      -x XMINMAX, --xminmax XMINMAX
                            the maximum of xmin.
      -C, --compare         compare with other two methods.

Two methods are compared. Equally spliation, and 1%, 10%, other.


### Example

	python main.py -i data/citation_list.txt -o output/grouping-result.png

	python main.py -i data/aps-citations.txt -o output/aps-grouping-result.png

### Format of citation_list.txt

    1,2,1,1,2,3,4,55,6,7,7,8 ...

Lines in citation_list.txt are citation count of each paper in a dataset separated by comma.

### Result format
The output is an image showing results of the citation distribution and the change rate of slope of citation distributions.
![Grouping results of the demo](output/wos_all_result.png)

### Comparation

| methods   |      xmin      |  xmax | lowly cited percentage | medium cited percentage | highly cited percentage|
|:----------:|:-------------:|:------:|:------:|:------:|:------:|
| our methods |  12 | 1954 |59.68%|40.30%|0.02%|
| equally splitation |    4   |   15 |33%|33%|33%|
| top N | 51 |    220 |90%|9%|1%|


### Dataset

  * ArtMiner
  * APS
  * WOS


### Citations
If you use the algorithms in this repo, please cite us as:
    to be added

