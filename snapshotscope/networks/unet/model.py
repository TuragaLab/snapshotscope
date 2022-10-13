"""
Taken from https://github.com/wolny/pytorch-3dunet

Original License:

MIT License

Copyright (c) 2018 Adrian Wolny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import importlib

import torch.nn as nn
import torch.nn.functional as F

import snapshotscope.networks.deconv as d
from snapshotscope.networks.unet.buildingblocks import (
    Decoder,
    DoubleConv,
    Encoder,
    ExtResNetBlock,
)


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        testing=False,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        **kwargs
    ):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,  # skip pooling in the firs encoder
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    padding=conv_padding,
                )
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(
                    f_maps[i - 1],
                    out_feature_num,
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    pool_kernel_size=pool_kernel_size,
                    padding=conv_padding,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        **kwargs
    ):
        super(UNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            **kwargs
        )


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        **kwargs
    ):
        super(ResidualUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ExtResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            **kwargs
        )


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=(1, 3, 3),
        pool_kernel_size=(1, 2, 2),
        conv_padding=1,
        **kwargs
    ):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        else:
            assert len(conv_padding) == 3
            assert conv_padding[0] == 0
        super(UNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
            conv_padding=conv_padding,
            **kwargs
        )


class FUNet3D(nn.Module):
    """
    Fourier UNet for 3D volume reconstruction.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_shape,
        num_planes,
        scale_factors,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="crb",
        num_groups=8,
        is_segmentation=False,
        testing=False,
        conv_kernel_size=(5, 5, 5),
        conv_padding=(2, 2, 2),
        **kwargs
    ):
        super().__init__()

        num_levels = len(scale_factors)
        self.testing = testing
        self.num_planes = num_planes

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        fourier_f_maps = [f * self.num_planes for f in f_maps]

        self.encoder = d.MultiscaleFourierConv2D(
            in_channels,
            fourier_f_maps,
            in_shape,
            stride=2,
            scale_factors=scale_factors,
        )
        self.do_encoder_relu = False
        self.do_encoder_batchnorm = False
        if "do_encoder_relu" in kwargs:
            self.do_encoder_relu = True
        if "do_encoder_batchnorm" in kwargs:
            self.encoder_batchnorms = nn.ModuleList(
                [
                    nn.BatchNorm2d(fourier_f_maps[l], momentum=0.1)
                    for l in range(num_levels)
                ]
            )
            self.do_encoder_batchnorm = True

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (1, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                scale_factor=(1, 2, 2),
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = list(reversed(self.encoder(x)))
        all_encoders_features = encoders_features

        # apply relu if specified
        if self.do_encoder_relu:
            encoders_features = [F.leaky_relu(ef) for ef in encoders_features]

        # apply batchnorm if specified
        if self.do_encoder_batchnorm:
            encoders_features = [
                bn(ef) for bn, ef in zip(self.encoder_batchnorms, encoders_features)
            ]

        # reshape features to 3d
        for scale_idx in range(len(encoders_features)):
            ef = encoders_features[scale_idx]
            shape = ef.shape
            encoders_features[scale_idx] = ef.view(
                -1,
                int(shape[1] / self.num_planes),
                self.num_planes,
                shape[2],
                shape[3],
            )

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # WARN(dip): we have to assign the first feature in the list to the
        # variable x explicitly because we removed the loop through encoders in
        # the original unet
        x = encoders_features[0]
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class FUNet2D(nn.Module):
    """
    Fourier UNet for 2D image reconstruction (e.g. RGB diffused/lensless camera
    images).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_shape,
        scale_factors,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="crb",
        num_groups=8,
        is_segmentation=False,
        testing=False,
        conv_kernel_size=(1, 3, 3),
        conv_padding=(0, 1, 1),
        **kwargs
    ):
        super().__init__()

        num_levels = len(scale_factors)
        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        fourier_f_maps = list(f_maps)

        self.encoder = d.MultiscaleFourierConv2D(
            in_channels,
            fourier_f_maps,
            in_shape,
            stride=2,
            scale_factors=scale_factors,
        )
        self.do_encoder_relu = False
        self.do_encoder_batchnorm = False
        if "do_encoder_relu" in kwargs:
            self.do_encoder_relu = True
        if "do_encoder_batchnorm" in kwargs:
            self.encoder_batchnorms = nn.ModuleList(
                [
                    nn.BatchNorm2d(fourier_f_maps[l], momentum=0.1)
                    for l in range(num_levels)
                ]
            )
            self.do_encoder_batchnorm = True

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (1, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                scale_factor=(1, 2, 2),
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = list(reversed(self.encoder(x)))
        all_encoders_features = encoders_features

        # apply relu if specified
        if self.do_encoder_relu:
            encoders_features = [F.leaky_relu(ef) for ef in encoders_features]

        # apply batchnorm if specified
        if self.do_encoder_batchnorm:
            encoders_features = [
                bn(ef) for bn, ef in zip(self.encoder_batchnorms, encoders_features)
            ]

        # add empty depth dimension to allow 3d convs
        for scale_idx in range(len(encoders_features)):
            ef = encoders_features[scale_idx]
            shape = ef.shape
            encoders_features[scale_idx] = ef.view(
                -1,
                shape[1],
                1,
                shape[2],
                shape[3],
            )

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # WARN(dip): we have to assign the first feature in the list to the
        # variable x explicitly because we removed the loop through encoders in
        # the original unet
        x = encoders_features[0]
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        # remove empty depth dimension
        x = x.squeeze(2)

        return x


class UNet3DVolumeReconstructor(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_planes,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="crb",
        num_groups=8,
        is_segmentation=False,
        testing=False,
        conv_kernel_size=(5, 5, 5),
        conv_padding=(2, 2, 2),
        pool_kernel_size=(1, 2, 2),
        **kwargs
    ):
        super().__init__()

        self.testing = testing
        self.num_planes = num_planes

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        encoder_f_maps = [f * self.num_planes for f in f_maps]

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(encoder_f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,  # skip pooling in the firs encoder
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    padding=conv_padding,
                )
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(
                    encoder_f_maps[i - 1],
                    out_feature_num,
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    pool_kernel_size=pool_kernel_size,
                    padding=conv_padding,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # reshape features to 3d
        x = x.view(
            -1,
            int(x.shape[1] / self.num_planes),
            self.num_planes,
            x.shape[3],
            x.shape[4],
        )
        for scale_idx in range(len(encoders_features)):
            ef = encoders_features[scale_idx]
            shape = ef.shape
            encoders_features[scale_idx] = ef.view(
                -1,
                int(shape[1] / self.num_planes),
                self.num_planes,
                shape[3],
                shape[4],
            )

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class ConvBlockYanny(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=True, activation='leakyrelu', transpose=False):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout3d(p=0.05)
            self.dropout2 = nn.Dropout3d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x


class UNet3DYanny(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu'):
        super().__init__()

        self.residual = residual

        if down == 'maxpool':
            self.down1 = nn.MaxPool3d(kernel_size=2)
            self.down2 = nn.MaxPool3d(kernel_size=2)
            self.down3 = nn.MaxPool3d(kernel_size=2)
            self.down4 = nn.MaxPool3d(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool3d(kernel_size=2)
            self.down2 = nn.AvgPool3d(kernel_size=2)
            self.down3 = nn.AvgPool3d(kernel_size=2)
            self.down4 = nn.AvgPool3d(kernel_size=2)
        elif down == 'conv':
            self.down1 = nn.Conv3d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv3d(128, 128, kernel_size=2, stride=2, groups=128)
            # make this padding (1, 0, 0) if not running on yanny dataset
            self.down4 = nn.Conv3d(256, 256, kernel_size=2, stride=2, padding=(1, 0, 0), groups=256)

            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == 'tconv':
            # make this kernel size (1, 2, 2) if not running on yanny dataset
            self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=(1, 2, 2), stride=2, groups=256)
            # make this output padding (1, 0, 0) if running on 8 chunks (not yanny dataset)!
            self.up2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, output_padding=(1, 0, 0), groups=128)
            self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2, groups=32)

            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlockYanny(n_channel_in, 32, residual, activation)
        self.conv2 = ConvBlockYanny(32, 64, residual, activation)
        self.conv3 = ConvBlockYanny(64, 128, residual, activation)
        self.conv4 = ConvBlockYanny(128, 256, residual, activation)

        self.conv5 = ConvBlockYanny(256, 256, residual, activation)

        self.conv6 = ConvBlockYanny(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlockYanny(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlockYanny(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlockYanny(2 * 32, n_channel_out, residual, activation)

        if self.residual:
            self.convres = ConvBlockYanny(n_channel_in, n_channel_out, residual, activation)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)  
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)  
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)  
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)  
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]
