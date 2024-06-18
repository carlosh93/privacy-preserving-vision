import numpy as np
import torch
import abc
import torchvision
from numpy.fft import ifftshift
import torch.fft
import poppy
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def attach_summaries(gpu_rank, name, var, train_info, image=False, log_image=False,
                     experiment=None, video=False, only_scalar=False):
    if gpu_rank == 0 and train_info is not None and experiment is not None:
        epoch = train_info["epoch"]
        total_steps_per_epoch = train_info["total_steps_per_epoch"]
        step = train_info["step"]
        summary_step = epoch * total_steps_per_epoch + step
        if summary_step == 0 or (step % train_info["log_interval"] == train_info["log_interval"] - 1):
            if only_scalar:
                experiment.log_metric(name, var, step=summary_step+1)
                # summary.add_scalar(name, var, summary_step)
            else:
                if image and video:
                    raise Exception("image and video can't be True at the same time.")

                if image:
                    def show_image(data):
                        sizes = np.shape(data)
                        # Assuming HWC format
                        fig_size = [sizes[0] / 100, sizes[1] / 100]
                        fig_obj = plt.figure(figsize=fig_size)
                        ax = plt.Axes(fig_obj, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig_obj.add_axes(ax)
                        ax.imshow(data)
                        return fig_obj

                    tmp = var.cpu().detach()  # .numpy()
                    # fig = show_image(tmp/tmp.max())
                    experiment.log_image(name=name, image_data=tmp/tmp.max(), step=summary_step+1, overwrite=False)
                    # summary.add_figure(name, fig, summary_step)
                    # summary.add_image(name, var/var.max(), summary_step)
                    if log_image:
                        log_img = torch.log(var + 1e-12)
                        log_img -= torch.mean(log_img)
                        tmp = log_img.cpu().detach()  # .numpy()
                        # fig = show_image(tmp / tmp.max())
                        experiment.log_image(name=name, image_data=tmp/tmp.max(), step=summary_step+1, overwrite=False)
                        # summary.add_figure(name + "_log", fig, summary_step)
                        # summary.add_image(name + "_log", log_img / log_img.max(), summary_step)
                if video:
                    img_grid = torchvision.utils.make_grid(var, nrow=int(var.shape[0]/2))
                    tmp = img_grid.cpu().detach().numpy()
                    sizes = np.shape(tmp)
                    fig_size = [sizes[2] / 100, sizes[1] / 100]
                    fig = plt.figure(figsize=fig_size)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(np.transpose(tmp, (1, 2, 0)))
                    experiment.log_figure(figure_name=name, figure=fig, step=summary_step+1, overwrite=False)
                    # summary.add_figure(name, fig, summary_step)

                # summary.add_scalar(name + '_mean', torch.mean(var), summary_step)
                experiment.log_metric(name=name + '_max', value=torch.max(var), step=summary_step+1)
                # summary.add_scalar(name + '_max', torch.max(var), summary_step)
                experiment.log_metric(name=name + '_min', value=torch.min(var), step=summary_step+1)
                # summary.add_scalar(name + '_min', torch.min(var), summary_step)
                # experiment.log_metric(name + '_histogram', var, step=summary_step)
                # summary.add_histogram(name + '_histogram', var, summary_step)


def get_zernike_volume(resolution, n_terms, scale_factor=1e-6):
    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
    return zernike_volume * scale_factor


def compl_exp_tf(phase, dtype=torch.complex64):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = phase.type(torch.float64)
    return torch.add(torch.cos(phase).type(dtype), 1.j * torch.sin(phase).type(dtype))


def circular_aperture(input_field):
    input_shape = list(input_field.shape)
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2, -input_shape[2] // 2: input_shape[2] // 2].astype(
        np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = torch.tensor((r < max_val).astype(np.float64), device=input_field.device)
    return aperture * input_field


def propagate_fresnel(input_field,
                      distance,
                      sampling_interval,
                      wave_lengths):
    input_shape = list(input_field.shape)
    propagation = FresnelPropagation(input_shape,
                                     distance=distance,
                                     discretization_size=sampling_interval,
                                     wave_lengths=wave_lengths)
    return propagation(input_field)


def ifftshift2d(a_tensor):
    input_shape = list(a_tensor.shape)

    def get_mylist(axis):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        return mylist

    new_tensor = a_tensor
    new_tensor = new_tensor[:, get_mylist(1), ...]
    new_tensor = new_tensor[:, :, get_mylist(2), ...]
    return new_tensor


def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = list(input_filter.shape)

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = F.pad(input_filter, [0, 0, 0, 0, pad_left, pad_right, pad_top, pad_bottom])
    else:
        padded = input_filter

    padded = padded.permute(2, 0, 1, 3)
    padded = ifftshift2d(padded)
    padded = padded.permute(1, 2, 0, 3)

    # Take FFT
    tmp = padded.permute(2, 3, 0, 1)
    tmp = torch.fft.fftn(tmp.type(torch.complex64), dim=[-1, -2])
    return tmp.permute(2, 3, 0, 1)


def transp_fft2d(a_tensor, permute=False):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    if permute:
        a_tensor_apply = a_tensor.permute(0, 3, 1, 2)
    else:
        a_tensor_apply = a_tensor
    # FFT 2D operates on the two innermost (last two!) dimensions as tf.signal.fft2d in Tensorflow
    a_fft2d = torch.fft.fftn(a_tensor_apply, dim=[-1, -2])
    if permute:
        return a_fft2d.permute(0, 2, 3, 1)
    else:
        return a_fft2d


def transp_ifft2d(a_tensor, permute=False):
    if permute:
        a_tensor_apply = a_tensor.permute(0, 3, 1, 2)
    else:
        a_tensor_apply = a_tensor

    a_ifft2d_transp = torch.fft.ifftn(a_tensor_apply, dim=[-1, -2])

    if permute:
        # Transpose back to [batch_size, x, y, channels]
        return a_ifft2d_transp.permute(0, 2, 3, 1)
    else:
        return a_ifft2d_transp


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    """
    Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    """
    # refractive index difference
    delta_N = refractive_idcs.reshape([1, 1, 1, -1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])
    # phase delay indiced by height field
    phi = torch.tensor(wave_nos * delta_N, device=height_map.device) * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts


def get_intensities(input_field):
    return torch.square(torch.abs(input_field))


def least_common_multiple(a, b):
    return abs(a * b) / np.math.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = list(input_image.shape)
    input_image = input_image.type(torch.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        input_image = input_image.permute(0, 3, 1, 2)
        avg_pool = torch.nn.AvgPool2d(factor, stride=factor)
        output_img = avg_pool(input_image)
        output_img = output_img.permute(0, 2, 3, 1)
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a "
                "large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        input_image = input_image.permute(0, 3, 1, 2)
        tv_resize = torchvision.transforms.Resize(size=2 * [upsample_factor * target_side_length], interpolation=0)
        img_upsampled = tv_resize(input_image)
        # img_upsampled = tf.image.resize_nearest_neighbor(input_image,
        #                                                size=2 * [upsample_factor * target_side_length])

        avg_pool = torch.nn.AvgPool2d(upsample_factor, stride=upsample_factor)
        output_img = avg_pool(img_upsampled)
        output_img = output_img.permute(0, 2, 3, 1)

    return output_img


def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    """Performs a convolution of an image and a psf in frequency space.
    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    """
    # img = torch.tensor(img, dtype=torch.float32)
    # psf = torch.tensor(psf, dtype=torch.float32)

    img_shape = list(img.shape)
    output_img_shape = img_shape
    pad_top = pad_bottom = pad_left = pad_right = 0
    if not circular:
        target_side_length = 2 * img_shape[2]

        height_pad = (target_side_length - img_shape[2]) / 2
        width_pad = (target_side_length - img_shape[2]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        # img = F.pad(img, [0, 0, pad_left, pad_right, pad_top, pad_bottom])
        img = F.pad(img, [pad_left, pad_right, pad_top, pad_bottom])
        img_shape = list(img.shape)

    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[2:])  # img_shape[1:3]
        otf = otf.permute(2, 3, 0, 1)

    if adjoint:
        result = transp_ifft2d(img_fft * torch.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)
    result = torch.abs(result)  # .type(torch.float32)

    if not circular:
        # Original TF implementation doen't have this resize
        resize = torchvision.transforms.Resize(size=output_img_shape[2:], interpolation=0)
        result = result[:, :, pad_top+1:-pad_bottom, pad_left+1:-pad_right]
        result = resize(result)

    return result


def gaussian_noise(image, img_shape, stddev=0.001):
    dtype = image.dtype
    return image + torch.normal(0.0, round(stddev, 6), size=img_shape, device=image.device).type(dtype)


# Classes Definition

class Propagation(abc.ABC):
    def __init__(self,
                 input_shape,
                 distance,
                 discretization_size,
                 wave_lengths):
        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = discretization_size

    @abc.abstractmethod
    def _propagate(self, input_field):
        """Propagate an input field through the medium
        """

    def __call__(self, input_field):
        return self._propagate(input_field)


class FresnelPropagation(Propagation):
    def _propagate(self, input_field):
        _, M_orig, N_orig, _ = self.input_shape
        # zero padding.
        Mpad = M_orig // 4
        Npad = N_orig // 4
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad

        padded_input_field = F.pad(input_field, [0, 0, Npad, Npad, Mpad, Mpad])

        [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2]

        # Spatial frequency
        fx = x / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)

        # We create a non-trainable variable so that this computation can be reused
        # from call to call.
        if torch.is_tensor(self.distance):
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum)
            constant_exp_part_init = torch.tensor(tmp, dtype=torch.float64, device=input_field.device)

            constant_exponent_part = torch.nn.Parameter(constant_exp_part_init, requires_grad=False)

            H = compl_exp_tf(self.distance * constant_exponent_part)

        else:  # Save some memory
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum * self.distance)
            # constant_exp_part_init = tf.constant_initializer(tmp)
            # constant_exponent_part = tf.Variable(name="Fresnel_kernel_constant_exponent_part",
            #                                      initial_value=constant_exp_part_init,
            #                                      shape=padded_input_field.shape,
            #                                      dtype=tf.float64,
            #                                      trainable=False)
            expo_part = torch.tensor(tmp, dtype=torch.float64, device=input_field.device)
            H = compl_exp_tf(expo_part)

        objFT = transp_fft2d(padded_input_field, permute=True)
        out_field = transp_ifft2d(objFT * H, permute=True)

        return out_field[:, Mpad:-Mpad, Npad:-Npad, :]


class PhasePlate:
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None):
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        self._build()

    def _build(self):
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            """self.height_map += torch.tensor(np.random.uniform(low=-self.height_tolerance,
                                                              high=self.height_tolerance,
                                                              size=list(self.height_map.shape)),
                                            dtype=self.height_map.dtype, device=self.height_map.device)"""
            self.height_map += (-self.height_tolerance - self.height_tolerance)*torch.rand(list(self.height_map.shape),
                                                                                           dtype=self.height_map.dtype,
                                                                                           device=self.height_map.device
                                                                                           ) + self.height_tolerance

        self.phase_shifts = phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_idcs)

    def __call__(self, input_field):
        return torch.mul(input_field.to(self.phase_shifts.device), self.phase_shifts)