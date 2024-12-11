# BdDETR

# Dataset
Download the [Common Objects in Context 2017](https://cocodataset.org/#download) to `./BdViT`.
The code supports a datasets with the following directory structure:
```
BdDETR
└─ coco
   ├─ annotations
   ├─ test2017
   ├─ train2017
   └─ val2017
```

# Vector decomposition
The weights to which the vector decomposition is applied are placed in `weights/detr/`.
When applying vector decomposition, the convolutional and fully connected layers are defined as `nn.conv2d` and `nn.linear`.
Quantize bits and basis change `--qb`, and the minimum and maximum values ​​of the vector decomposition change `init_iter` and `max_iter` in `bdnn_module/Exhaustive_decomposer_numpy.py`.
```
cd BdDETR
python3.10 decomposition.py --weights ./weights/detr/detr-r50-e632da11.pth --qb 8
```

# Decomposed weights
The decomposed weights can be downloaded [here](https://drive.google.com/file/d/1D2mto8ptMfxzuMQRnPMJbHks9vU3dQLU/view?usp=sharing).

# Evaluation
For evaluation, the model definition directory is `models_bd`.
If you change the comment out of `conv2d_binary` and `linear_binary_KA` in `bdnn_module/binary_functional.py`, you can choose DETR and BdDETR.
Quantize bits and basis change `--qb`.
```
cd BdDETR
python3.10 test.py --batch_size 2 --no_aux_loss --eval --coco_path coco --weights ./weights/detr/detr-r50-e632da11.pth --qb 8
```

# bdnn_module
```
build, binaryfunc_cython.c                         - Folders/files generated when compiling cython
binaryfunc_cython.cpython-36m-x86_64-linux-gnu.so  - Logical operation module for python3.6
binaryfunc_cython.cpython-37m-x86_64-linux-gnu.so  - Logical operation module for python3.7
binaryfunc_cython.cpython-38m-x86_64-linux-gnu.so  - Logical operation module for python3.8
binaryfunc_cython.cpython-39m-x86_64-linux-gnu.so  - Logical operation module for python3.9
binaryfunc_cython.cpython-310m-x86_64-linux-gnu.so - Logical operation module for python3.10
binaryfunc_cython.pyx                              - cython logic operation definition file
Exhaustive_decomposer_numpy.py                     - Vector decomposition module for Exhaustive algorithm
conv_binary.py                                     - BdDETR Convolution layer definition file
fully_connected_binary.py                          - BdDETR FullyConnected layer definition file
setup.py                                           - Program to compile cython code
utils 
  |---calc_comput_complex.py                       - Feature map size calculation module
  |---decomposition.py                             - Module to select method and perform decomposition
  |---extract_weight_ext.py                        - Module to extract parameters and simultaneously perform analysis such as standard deviation
```
