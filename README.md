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
The output is an image showing results of the citation distribution and the change rate of slope of citation distributions. The result of adjusted R square is recommended to adopt, but for some datasets, the other two R squares may also be useful. The format could be png, pdf, jpg, etc.
![Grouping results of the demo](output/aps-grouping-result.png)

### Comparation

| methods   |      xmin      |  xmax | lowly cited percentage | medium cited percentage | highly cited percentage|
|:----------:|:-------------:|:------:|:------:|:------:|:------:|
| our methods |  9 | 315 |60.27%|39.55%|0.17%|
| equally splitation |    4   |   10 |33%|33%|33%|
| top N | 30 |    120 |90%|9%|1%|


### Dataset
Three datasets are utilized in this paper: ArnetMiner and Microsoft Academic Graph (MAG) for the field of computer science, as well as American Physical Society (APS) for the field of physics. ArnetMiner covers important conferences and journals from the domain of computer science. The citation network of ArnetMiner comprises 1,286,254 papers/references ranging from 1936 to 2014 and 8,024,869 local citation relationships. Among all of the papers, 940,198 papers have received at least one citation. Regarding the MAG dataset, only papers in the computer science field are adopted in this paper (MAG-CS); the number of papers in MAG-CS is 5,249,815. We build a citation network composed of 9,703,104 papers and 44,944,243 citation relations. Out of all of these papers, 2,584,967 have been cited at least once. As for the APS dataset, it covers the bibliographic metadata of 13 top journals in physics, including 596,786 publications and 7,211,978 citation relationships among them, in which there are 515,867 publications that have received at least one citation.

### Citations
If you use the algorithms in this repo, please cite us as:
    to be added

