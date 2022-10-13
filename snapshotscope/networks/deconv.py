from fouriernet import *
import torch.nn as nn

from snapshotscope.utils import *
from snapshotscope.utils import _list, _pair, _triple

from .unet import FUNet3D, UNet3DVolumeReconstructor, UNet2D as UNet2DVolumeReconstructor


class MultiscaleFourierConv2D(nn.Module):
    """
    Applies a 2D Fourier convolution over an input signal composed of several
    input planes. That is, 2D convolution is performed by performing a Fourier
    transform, multiplying a kernel and the input, then taking the inverse
    Fourier transform of the result. This layer also supports efficient
    multiscale feature output by simply cropping the calculated Fourier
    features to the desired resolutions.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_shape,
        stride=1,
        padding=True,
        bias=True,
        scale_factors=1,
        reduce_channels=True,
        real_feats_mode="index",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = _list(out_channels)
        in_shape = nn.modules.utils._pair(in_shape)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = padding
        # kernel size depends on whether we pad or not
        if self.padding and self.stride == (1, 1):
            self.in_shape = (in_shape[0] * 2 - 1, in_shape[1] * 2 - 1)
        else:
            # TODO(dd): handle arbitrary strides or make strides a boolean
            self.in_shape = in_shape
        self.scale_factors = _list(scale_factors)
        weights = []
        # these scale factors are absolute, meaning they define the scale
        # factor from the top level
        for scale_idx, s in enumerate(self.scale_factors):
            in_shape = tuple(
                int(self.in_shape[d] / s) for d in range(len(self.in_shape))
            )
            weights.append(
                nn.Parameter(
                    torch.Tensor(out_channels[scale_idx], in_channels, *in_shape, 2)
                )
            )
        self.weights = nn.ParameterList(weights)
        self.reduce_channels = reduce_channels
        self.out_shapes = [
            tuple(int(self.in_shape[d] / s) for d in range(len(self.in_shape)))
            for s in self.scale_factors
        ]
        if bias and self.reduce_channels:
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.Tensor(n, 1, 1)) for n in self.out_channels]
            )
        elif bias:
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.Tensor(n, 1, 1, 1)) for n in self.out_channels]
            )
        else:
            self.register_parameter("biases", None)
        self.real_feats_mode = real_feats_mode
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for weight in self.weights:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.biases is not None:
            for weight, bias in zip(self.weights, self.biases):
                fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, in_shape={in_shape}, scale_factors={scale_factors}, out_shapes={out_shapes}"
        if self.biases is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def forward(self, im):
        # determine input, output, and fourier domain sizes
        insize = im.size()
        outsize = (self.out_channels, im.size(-2), im.size(-1))
        fftsize = [insize[-2] + outsize[-2] - 1, insize[-1] + outsize[-1] - 1]

        # construct zero pad for real signal if needed
        if self.padding:
            pad_size = (
                0,
                1,
                0,
                fftsize[-1] - insize[-1],
                0,
                fftsize[-2] - insize[-2],
            )
        else:
            pad_size = (0, 1)
        # add channel for imaginary component and perform FFT
        fourier_im = torch.fft(F.pad(im.unsqueeze(-1), pad_size), 2)

        if self.stride != (1, 1):
            # skip inputs if stride > 1
            stride = self.stride
            fourier_im = fourier_im[:, :, :: stride[-2], :: stride[-1], :]

        # calculate features in fourier space
        # (add dimension for batch broadcasting)
        fourier_feats = []
        for scale_idx, s in enumerate(self.scale_factors):
            out_shape = self.out_shapes[scale_idx]
            cropped_im = center_crop2d_complex(fourier_im, out_shape)
            fourier_feats.append(
                cmul2(cropped_im.unsqueeze(1), self.weights[scale_idx])
            )

        # retrieve real component of signal
        real_feats = []
        for scale_idx, ff in enumerate(fourier_feats):
            rf = torch.ifft(ff, 2)
            if self.real_feats_mode == "index":
                indices = torch.tensor(0, device=fourier_im.device)
                rf = rf.index_select(-1, indices)
                rf = rf.squeeze(-1)
            elif self.real_feats_mode == "abs":
                rf = cabs(rf)
            else:
                return NotImplemented

            # crop feature maps back to original size if we padded
            if self.padding and self.stride == (1, 1):
                cropsize = [fftsize[-2] - insize[-2], fftsize[1] - insize[-1]]
                cropsize_left = [int(c / 2) for c in cropsize]
                cropsize_right = [int((c + 1) / 2) for c in cropsize]
                rf = F.pad(
                    rf,
                    (
                        -cropsize_left[-1],
                        -cropsize_right[-1],
                        -cropsize_left[-2],
                        -cropsize_right[-2],
                    ),
                )

            # sum over input channels to get correct number of output channels
            if self.reduce_channels:
                rf = rf.sum(2)

            # add bias term
            if self.biases is not None:
                rf = rf + self.biases[scale_idx]

            # append to real feature list
            real_feats.append(rf)

        return real_feats


class FourierConv2D(nn.Module):
    """
    Applies a 2D Fourier convolution over an input signal composed of several
    input planes. That is, 2D convolution is performed by performing a Fourier
    transform, multiplying a kernel and the input, then taking the inverse
    Fourier transform of the result.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=True,
        bias=True,
        reduce_channels=True,
        real_feats_mode="index",
    ):
        super(FourierConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = padding
        # kernel size depends on whether we pad or not
        if self.padding and self.stride == (1, 1):
            self.kernel_size = (kernel_size[0] * 2 - 1, kernel_size[1] * 2 - 1)
        else:
            self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size, 2)
        )
        self.reduce_channels = reduce_channels
        if bias and self.reduce_channels:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1))
        else:
            self.register_parameter("bias", None)
        self.real_feats_mode = real_feats_mode
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def forward(self, im):
        # determine input, output, and fourier domain sizes
        insize = im.size()
        outsize = (self.out_channels, im.size(-2), im.size(-1))
        fftsize = [insize[-2] + outsize[-2] - 1, insize[-1] + outsize[-1] - 1]

        # construct zero pad for real signal if needed
        if self.padding:
            pad_size = (
                0,
                1,
                0,
                fftsize[-1] - insize[-1],
                0,
                fftsize[-2] - insize[-2],
            )
        else:
            pad_size = (0, 1)
        # add channel for imaginary component and perform FFT
        fourier_im = torch.fft(F.pad(im.unsqueeze(-1), pad_size), 2)

        if self.stride != (1, 1):
            # skip inputs if stride > 1
            stride = self.stride
            fourier_im = fourier_im[:, :, :: stride[-2], :: stride[-1], :]

        # calculate features in fourier space
        # (add dimension for batch broadcasting)
        fourier_feats = cmul2(fourier_im.unsqueeze(1), self.weight)

        # retrieve real component of signal
        if self.real_feats_mode == "index":
            indices = torch.tensor(0, device=fourier_im.device)
            real_feats = torch.ifft(fourier_feats, 2).index_select(-1, indices)
            real_feats = real_feats.squeeze(-1)
        elif self.real_feats_mode == "abs":
            real_feats = cabs(torch.ifft(fourier_feats, 2))
        else:
            return NotImplemented

        # crop feature maps back to original size if we padded
        if self.padding and self.stride == (1, 1):
            cropsize = [fftsize[-2] - insize[-2], fftsize[1] - insize[-1]]
            cropsize_left = [int(c / 2) for c in cropsize]
            cropsize_right = [int((c + 1) / 2) for c in cropsize]
            real_feats = F.pad(
                real_feats,
                (
                    -cropsize_left[-1],
                    -cropsize_right[-1],
                    -cropsize_left[-2],
                    -cropsize_right[-2],
                ),
            )

        # sum over input channels to get correct number of output channels
        if self.reduce_channels:
            real_feats = real_feats.sum(2)

        # add bias term
        real_feats = real_feats + self.bias

        return real_feats


class WienerFilter(nn.Module):
    def __init__(
            self,
            sigma_squared: float = 0.1,
            trainable_filter: bool = False,
            filter_init: torch.Tensor = None
    ):
        super().__init__()
        self.sigma_squared = nn.Parameter(torch.tensor(sigma_squared, dtype=torch.float32))
        self.trainable_filter = trainable_filter
        if trainable_filter:
            self.psf = nn.Parameter(filter_init)

    def forward(self, im: torch.Tensor, psf: torch.Tensor=None) -> torch.Tensor:
        if self.trainable_filter:
            psf = self.psf
        # (N, C, H, W) -> (1, H, W), N, C = 1
        im = torch.squeeze(im).unsqueeze(0)
        # (1, H, W) -> (D, H, W), D = depth of PSF
        im = im.expand_as(psf)
        image_shape = im.shape
        psf_shape = psf.shape
        fft_shape = [image_shape[-2]+psf_shape[-2]-1, image_shape[-1]+psf_shape[-1]-1]

        # zero pad real signal and add a channel for the imaginary part
        fft_im = torch.fft(
            F.pad(im.unsqueeze(-1),(0,1,0,fft_shape[-1]-image_shape[-1],0,fft_shape[-2]-image_shape[-2]))
            ,2)

        fft_psf = torch.fft(
            F.pad(psf.unsqueeze(-1),(0,1,0,fft_shape[-1]-psf_shape[-1],0,fft_shape[-2]-psf_shape[-2]))
            ,2)

        wiener_numerator = cmulconjy(fft_im, fft_psf)
        safe_regularizer = F.relu(self.sigma_squared) + 1e-6
        wiener_denominator = real_to_complex(fft_psf.pow(2).sum(-1) + safe_regularizer)
        fourier_reconstruction = cdiv(wiener_numerator, wiener_denominator)
        indices = torch.tensor(0, device=psf.device)
        reconstruction = torch.ifft(fourier_reconstruction, 2).index_select(-1, indices)
        reconstruction = reconstruction.squeeze(-1)

        cropsize = [fft_shape[-2] - psf_shape[-2], fft_shape[-1] - psf_shape[-1]]
        cropsize_left = [int(c / 2) for c in cropsize]
        cropsize_right = [int((c + 1) / 2) for c in cropsize]
        reconstruction = F.pad(
            reconstruction,
            (
                -cropsize_left[-1],
                -cropsize_right[-1],
                -cropsize_left[-2],
                -cropsize_right[-2],
            ),
        )
        reconstruction = reconstruction.unsqueeze(0) # (D, H, W) -> (N=1, C, H, W), C == D

        return reconstruction


class Reshape2D3D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, tensor_2d):
        return tensor_2d.view(
            -1,
            int(tensor_2d.shape[1] / self.d),
            self.d,
            tensor_2d.shape[2],
            tensor_2d.shape[3],
        )


class InputScalingSequential(nn.Sequential):
    """
    Extension of Sequential that divides the input by a specified
    quantile value of the input (e.g. 0.99 quantile). At the end of
    the sequential calls to all the modules, the recorded scale is
    multiplied back into the output. This is a way to stabilize
    network training over inputs of drastically changing scales
    without using BatchNorm. The advantage is that the network doesn't
    need to understand exactly how to scale the output, e.g. if an
    input has 10x larger values, the output will also have 10x larger
    values.
    """

    def __init__(self, quantile, quantile_scale, *args):
        super().__init__(*args)
        self.quantile = quantile
        self.quantile_scale = quantile_scale

    def forward(self, input):
        # unscale input by quantile and record quantile scale
        quantile = torch.quantile(input, self.quantile).detach()
        scale = self.quantile_scale * quantile
        input = input / scale
        for module in self:
            input = module(input)
        # rescale output by recorded scale
        input = input * scale
        return input


class InputNormingSequential(nn.Sequential):
    """
    Extension of Sequential that shifts and scales the input to be
    zero mean and unit variance. At the end of the sequential calls to
    all the modules, the recorded scale and shift are undone for the
    final output. This is a way to stabilize network training over
    inputs of drastically changing scales without using BatchNorm. The
    advantage is that the network doesn't need to understand exactly
    how to scale the output, e.g. if an input has 10x larger values,
    the output will also have 10x larger values.
    """

    def __init__(self, dim, *args):
        super().__init__(*args)
        self.dim = dim

    def forward(self, input):
        # shift sample to zero mean and record shift
        if self.dim is None:
            shift = torch.mean(input).detach()
        else:
            shift = torch.mean(input, self.dim).detach()
        input = input - shift.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        # scale sample to unit variance and record scale
        if self.dim is None:
            scale = torch.sqrt(torch.var(input).detach() + 1e-5)
        else:
            scale = torch.sqrt(
                torch.var(input, self.dim, unbiased=False).detach() + 1e-5
            )
        input = input / scale.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        for module in self:
            input = module(input)
        # rescale output by recorded scale
        input = input * scale.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        # reshift output by recorded shift
        input = input + shift.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        return input


def FourierNet2D(
    fourier_out,
    fourier_kernel_size,
    num_planes,
    fourier_conv_args=None,
    conv_kernel_sizes=[11],
    conv_fmap_nums=None,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    fourierbn_momentum=0.1,
    convbn_momentum=0.1,
    device="cpu",
):
    # ensure kernel sizes and fmap nums are lists
    conv_kernel_sizes = _list(conv_kernel_sizes)
    if conv_fmap_nums is None:
        conv_fmap_nums = [num_planes] * len(conv_kernel_sizes)
    assert conv_fmap_nums[-1] == num_planes, "Must output number of planes"
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(1, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    if fourier_conv_args is None:
        fourier_conv_args = {}
    # create fourier convolution and batch norm
    fourier_conv = FourierConv2D(
        1, fourier_out, fourier_kernel_size, **fourier_conv_args
    ).to(device)
    fourier_relu = nn.LeakyReLU().to(device)
    fourierbn = nn.BatchNorm2d(fourier_out, momentum=fourierbn_momentum).to(device)
    layers += [
        ("fourier_conv", fourier_conv),
        ("fourier_relu", fourier_relu),
        ("fourier_bn", fourierbn),
    ]
    # create convolution layers
    # if multiple convolution layers, use LeakyReLU until last layer
    # use batch norms between convolutions (after activation)
    # ReLU on final layer for non-negative output, no batch norm at end
    for i in range(len(conv_fmap_nums)):
        previous_fmap_nums = fourier_out if i == 0 else conv_fmap_nums[i - 1]
        conv = nn.Conv2d(
            previous_fmap_nums,
            conv_fmap_nums[i],
            conv_kernel_sizes[i],
            padding=int(math.floor(conv_kernel_sizes[i] / 2)),
        )
        conv = conv.to(device)
        layers.append((f"conv2d_{i+1}", conv))
        if i < len(conv_fmap_nums) - 1:
            conv_relu = nn.LeakyReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            conv_bn = nn.BatchNorm2d(conv_fmap_nums[i], momentum=convbn_momentum).to(
                device
            )
            layers.append((f"conv{i+1}_bn", conv_bn))
        else:
            conv_relu = nn.ReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(
            quantile, quantile_scale, OrderedDict(layers)
        )
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    else:
        reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


def FourierNet3D(
    fourier_out,
    fourier_kernel_size,
    num_planes,
    fourier_conv_args=None,
    conv_kernel_sizes=[11],
    conv_fmap_nums=None,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    fourierbn_momentum=0.1,
    convbn_momentum=0.1,
    device="cpu",
):
    # ensure kernel sizes and fmap nums are lists
    conv_kernel_sizes = _list(conv_kernel_sizes)
    conv_kernel_sizes = [_triple(k) for k in conv_kernel_sizes]
    if conv_fmap_nums is None:
        conv_fmap_nums = [num_planes] * len(conv_kernel_sizes)
    assert conv_fmap_nums[-1] == 1, "Must output single feature"
    assert (
        fourier_out % num_planes == 0
    ), "Fourier feature map number must be multiple of number of planes"
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(1, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    if fourier_conv_args is None:
        fourier_conv_args = {}
    # create fourier convolution and batch norm
    fourier_conv = FourierConv2D(
        1, fourier_out, fourier_kernel_size, **fourier_conv_args
    ).to(device)
    fourier_relu = nn.LeakyReLU().to(device)
    fourierbn = nn.BatchNorm2d(fourier_out, momentum=fourierbn_momentum).to(device)
    reshape2d3d = Reshape2D3D(num_planes)
    layers += [
        ("fourier_conv", fourier_conv),
        ("fourier_relu", fourier_relu),
        ("fourier_bn", fourierbn),
        ("fourier_reshape_2d_3d", reshape2d3d),
    ]
    # create convolution layers
    # if multiple convolution layers, use LeakyReLU until last layer
    # use batch norms between convolutions (after activation)
    # ReLU on final layer for non-negative output, no batch norm at end
    for i in range(len(conv_fmap_nums)):
        previous_fmap_nums = (
            int(fourier_out / num_planes) if i == 0 else conv_fmap_nums[i - 1]
        )
        conv = nn.Conv3d(
            previous_fmap_nums,
            conv_fmap_nums[i],
            conv_kernel_sizes[i],
            padding=[int(math.floor(k / 2)) for k in conv_kernel_sizes[i]],
        )
        conv = conv.to(device)
        layers.append((f"conv3d_{i+1}", conv))
        if i < len(conv_fmap_nums) - 1:
            conv_relu = nn.LeakyReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            conv_bn = nn.BatchNorm3d(conv_fmap_nums[i], momentum=convbn_momentum).to(
                device
            )
            layers.append((f"conv{i+1}_bn", conv_bn))
        else:
            conv_relu = nn.ReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(
            quantile, quantile_scale, OrderedDict(layers)
        )
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    else:
        reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


def FourierUNet3D(
    in_shape,
    num_planes,
    scale_factors=[1, 2, 4, 8],
    funet_fmaps=[12, 12, 12, 12],
    conv_kernel_size=(3, 5, 5),
    conv_padding=(1, 2, 2),
    funet_kwargs={},
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(1, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create funet
    funet = FUNet3D(
        1,
        1,
        in_shape,
        num_planes,
        scale_factors,
        f_maps=funet_fmaps,
        conv_kernel_size=conv_kernel_size,
        conv_padding=conv_padding,
        **funet_kwargs,
    ).to(device)
    funet_relu = nn.ReLU().to(device)
    layers += [("funet", funet), ("funet_relu", funet_relu)]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(
            quantile, quantile_scale, OrderedDict(layers)
        )
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    else:
        reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


def FourierNetRGB(
    fourier_out,
    fourier_kernel_size,
    fourier_conv_args=None,
    conv_kernel_sizes=[11],
    conv_fmap_nums=None,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    fourierbn_momentum=0.1,
    convbn_momentum=0.1,
    device="cpu",
):
    # ensure kernel sizes and fmap nums are lists
    conv_kernel_sizes = _list(conv_kernel_sizes)
    conv_kernel_sizes = [_pair(k) for k in conv_kernel_sizes]
    assert conv_fmap_nums[-1] == 3, "Must output 3 features (RGB)"
    # create image batch norm
    if input_scaling_mode is None or input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(3, momentum=imbn_momentum).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of None, 'scaling', 'norming', or 'batchnorm'"
        )
    if fourier_conv_args is None:
        fourier_conv_args = {}
    # create fourier convolution and batch norm
    fourier_conv = FourierConv2D(
        3, fourier_out, fourier_kernel_size, **fourier_conv_args
    ).to(device)
    fourier_relu = nn.LeakyReLU().to(device)
    fourierbn = nn.BatchNorm2d(fourier_out, momentum=fourierbn_momentum).to(device)
    layers += [
        ("fourier_conv", fourier_conv),
        ("fourier_relu", fourier_relu),
        ("fourier_bn", fourierbn),
    ]
    # create convolution layers
    # if multiple convolution layers, use LeakyReLU until last layer
    # use batch norms between convolutions (after activation)
    # ReLU on final layer for non-negative output, no batch norm at end
    for i in range(len(conv_fmap_nums)):
        previous_fmap_nums = (
            fourier_out if i == 0 else conv_fmap_nums[i - 1]
        )
        conv = nn.Conv2d(
            previous_fmap_nums,
            conv_fmap_nums[i],
            conv_kernel_sizes[i],
            padding=[int(math.floor(k / 2)) for k in conv_kernel_sizes[i]],
        )
        conv = conv.to(device)
        layers.append((f"conv2d_{i+1}", conv))
        if i < len(conv_fmap_nums) - 1:
            conv_relu = nn.LeakyReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            conv_bn = nn.BatchNorm2d(conv_fmap_nums[i], momentum=convbn_momentum).to(
                device
            )
            layers.append((f"conv{i+1}_bn", conv_bn))
        else:
            conv_relu = nn.ReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(
            quantile, quantile_scale, OrderedDict(layers)
        )
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
    else:
        reconstruct = nn.Sequential(OrderedDict(layers))
    return reconstruct


def WienerUNet3D(
    num_planes,
    trainable_filter=False,
    filter_init=None,
    sigma_squared=0.1,
    input_scaling_mode="scaling",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    wienerbn_momentum=0.1,
    device="cpu",
    **kwargs,
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        return NotImplemented
        # imbn = nn.BatchNorm3d(1, momentum=imbn_momentum).to(device)
        # layers = [imbn]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create wiener filter layer
    layers += [
        WienerFilter(sigma_squared, trainable_filter=trainable_filter, filter_init=filter_init).to(device),
        Reshape2D3D(num_planes),
        # nn.BatchNorm3d(1, momentum=wienerbn_momentum).to(device)
    ]
    # create unet layers
    layers += [
        UNet3DYanny().to(device),
        # nn.ReLU().to(device),
    ]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(quantile, quantile_scale, *layers)
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3, 4), *layers)
    else:
        reconstruct = nn.Sequential(*layers)
    return reconstruct


def UNet2D(
    in_channels,
    out_channels,
    f_maps=64,
    layer_order="crb",
    num_levels=4,
    conv_padding=1,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
    **kwargs,
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        if not kwargs['imbn_momentum']:
            imbn_momentum = 0.1
        imbn = nn.BatchNorm3d(1, momentum=imbn_momentum).to(device)
        layers = [imbn]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create unet layers
    layers += [
        UNet2DVolumeReconstructor(
            in_channels,
            out_channels,
            final_sigmoid=False,
            f_maps=f_maps,
            layer_order=layer_order,
            num_levels=num_levels,
            is_segmentation=False,
            conv_padding=conv_padding,
            **kwargs,
        ).to(device),
        nn.ReLU().to(device),
    ]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(quantile, quantile_scale, *layers)
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3, 4), *layers)
    else:
        reconstruct = nn.Sequential(*layers)
    return reconstruct


def UNet3D(
    in_channels,
    out_channels,
    num_planes,
    f_maps=64,
    layer_order="crb",
    num_levels=4,
    conv_padding=1,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
    **kwargs,
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        if not kwargs['imbn_momentum']:
            imbn_momentum = 0.1
        imbn = nn.BatchNorm3d(1, momentum=imbn_momentum).to(device)
        layers = [imbn]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create unet layers
    layers += [
        UNet3DVolumeReconstructor(
            in_channels,
            out_channels,
            num_planes,
            final_sigmoid=False,
            f_maps=f_maps,
            layer_order=layer_order,
            num_levels=num_levels,
            is_segmentation=False,
            conv_padding=conv_padding,
            **kwargs,
        ).to(device),
        nn.ReLU().to(device),
    ]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(quantile, quantile_scale, *layers)
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3, 4), *layers)
    else:
        reconstruct = nn.Sequential(*layers)
    return reconstruct

