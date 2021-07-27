# Fourier Convolutions

This repository contains official PyTorch code for implementing Fourier convolutions and FourierNets/FourierUNets from the paper: [*Programmable 3D snapshot microscopy with Fourier convolutional networks*](https://arxiv.org/abs/2104.10611).

![Figure 1 from the paper showing our FourierNet/FourierUNet architectures](figs/fig1.png)
![Figure 2 from the paper showing how FourierNet succeeds at optimizing microscopes](figs/fig2.png)
![Figure 4 from the paper showing how FourierNet beats state of the art reconstruction algorithms for computational photography](figs/fig4.png)

What is included:

* Implementations of FourierNet/FourierUNet architectures from the paper.
* Scripts to recreate experiments from the paper.
* The simulation code required to run the experiments.

What is **not** included:
* This repository does **not** include the data required to run the experiments. The data can be obtained from [Figshare](https://figshare.com) (coming soon).

# Installation

We have tested `snapshotscope` on Python 3.7 with PyTorch 1.7. Newer versions of PyTorch will remove the old FFT interface, and cause this software to fail.

To install the library (required for running the experiment scripts), you can run:

```
$ pip install git+https://github.com/TuragaLab/snapshotscope
```

# Usage

```
$ python exp.py <train | test>
```

The experiment scripts in `experiments` follow the same pattern: they take one argument, which is whether to train or test. If you want to modify any other aspect of the training, you can simply change those settings in the corresponding `exp.py` file.
