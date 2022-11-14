# Differentiable 3D Snapshot Microscope

This repository contains official PyTorch code for implementing a differentiable snapshot microscope and the relevant experiment scripts from the paper: [*FourierNets enable the design of highly non-local optical encoders for computational imaging*](https://arxiv.org/abs/2104.10611).

![Figure 1 from the paper showing our FourierNet/FourierUNet architectures](figs/fig1.png)
![Figure 2 from the paper showing how FourierNet succeeds at optimizing microscopes](figs/fig2.png)
![Figure 4 from the paper showing how FourierNet beats state of the art reconstruction algorithms for computational photography](figs/fig4.png)

What is included:

* A differentiable simulation of a 3D snapshot microscope.
* Scripts to recreate experiments from the paper, both using the microscope and using DLMD for computational photography reconstruction.
* Implementations of FourierNet/FourierUNet architectures from the paper. This is a dependency for running the experiments, but not for using the simulation. These are included in this repository for ease of replication. If you just want implementations of FourierNets, you can obtain implementations for both PyTorch and JAX/Flax from [TuragaLab/fouriernet](https://github.com/TuragaLab/fouriernet) (**coming soon**).

What is **not** included:
* This repository does **not** include the data required to run the experiments. The data can be obtained from [Figshare](https://figshare.com) (**coming soon**).

# Installation

We have tested `snapshotscope` on Python 3.7 with PyTorch 1.7. **Newer versions of PyTorch are not supported because they have swapped to a new FFT interface.**

To install the library (required for running the experiment scripts), you can run:

```
$ pip install git+https://github.com/TuragaLab/snapshotscope
```

# Usage

```
$ python exp.py <train | test>
```

**You will probably want to change the locations where data is read and results are saved by modifying the provided exp.py files.** These experiment scripts in `experiments` follow the same pattern: they take one argument, which is whether to train or test. If you want to modify any other aspect of the training, you can simply change those settings in the corresponding `exp.py` file.
