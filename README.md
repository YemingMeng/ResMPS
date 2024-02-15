# ResMPS: Residual Matrix Product State

ResMPS (short for 'Residual Matrix Product State') is a PyTorch based tensor network machine learning architecture.

The idea of ResMPS is inpired by residual networks, and outperforms the state-of-the-art tensor network models in terms of efficiency, stability, and expression power.

For further details, please see our paper [SciPost Physics 14.6 (2023): 142](https://scipost.org/SciPostPhys.14.6.142).

## Installation
1. Install dependencies.
  - torch
  - torchvision
  - hilbertcurve
  - matplotlib
  - sklearn
  - dill

The `cuda` version of `PyTorch` is recommended.

2. Clone this repo.
```bash
git clone https://github.com/YemingMeng/ResMPS.git
```

## Usage/Examples
- Print help of usage information.
```bash
python main.py -h
```
- Typical command example, here `cuda` for `GPU` acceleration and `fashion_mnist` is the dataset.
```bash
python main.py --device cuda --dataset fashion_mnist
```
- To reproduce results in [arxiv:2012.11841](https://scipost.org/submissions/2012.11841/), run
```bash
python examples.py
```
- Use ResMPS in general case, create your python script like this
```
from ResMPS import ResMPS

nfeatures = 2     # dimension of the feature map function
nchi      = 12    # dimension of the virtual feature
nrow      = 28    # number of rows of the input image
ncol      = 28    # number of columns of the input image
nlength   = 785   # length of ResMPS, needs to match the dimension of input
noutput   = 10    # dimension of the output, i.e., the total number of categories

batch_size = 1000 # input batch size
cf = ResMPS(nfeatures, nchi, nlength, noutput) # create an instance of ResMPS
input = th.rand(batch_size, 1, nrow, ncol, device=cf.device) # generate random input data
output = cf(input)  # data processing by using ResMPS
print(output.shape) # check the shape of output
```

## Citing ResMPS
Include this BibTex Entry to your `.bib` file
```
@misc{https://doi.org/10.48550/arxiv.2012.11841,
  doi = {10.48550/ARXIV.2012.11841},
  url = {https://arxiv.org/abs/2012.11841},
  author = {Meng, Ye-Ming and Zhang, Jing and Zhang, Peng and Gao, Chao and Ran, Shi-Ju},
  title = {Residual Matrix Product State for Machine Learning},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)