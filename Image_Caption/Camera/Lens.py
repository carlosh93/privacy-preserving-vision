from torch import nn
import os
from .Utils import *
import torchvision
import cv2
from PIL import Image

device = torch.device("cuda")


class OpticsZernike(nn.Module):
    def __init__(self,
                 input_shape,
                 device,
                 experiment=None,  # comet.ml experiment
                 sensor_distance=25e-3,
                 refractive_idcs=np.array([1.499, 1.493, 1.488]),
                 wave_lengths=np.array([460, 550, 640]) * 1e-9,
                 height_tolerance=20e-9,
                 wave_resolution=(736, 736),
                 patch_size=368,
                 sample_interval=2e-6,
                 upsample=False,
                 frames=8,
                 optics_cfg=1,
                 zernike_terms=350,
                 mask_1=None,
                 mask_2=None):
        super(OpticsZernike, self).__init__()
        # self.output_dim = output_dim
        self.device = device
        self.sensor_distance = sensor_distance  # Distance of sensor to aperture
        self.height_tolerance = height_tolerance  # manufacturing error
        self.upsample = upsample
        self.patch_size = patch_size  # Size of patches to be extracted from images, and resolution of simulated sensor
        self.wave_lengths = wave_lengths  # Wave lengths to be modeled and optimized for
        self.sample_interval = sample_interval  # Sampling interval (size of one "pixel" in the simulated wavefront)
        self.refractive_idcs = refractive_idcs  # Refractive idcs of the phaseplate
        self.frames = frames
        self.optics_cfg = optics_cfg
        self.zernike_terms = zernike_terms
        self.mask_1 = mask_1
        self.mask_2 = mask_2

        if wave_resolution is None:
            self.wave_res = [patch_size * 4, patch_size * 4]
        else:
            self.wave_res = wave_resolution

        self.physical_size = float(self.wave_res[0] * self.sample_interval)
        if self.device == torch.device(0):
            print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_res[0]))

        # Declare Zernike Volume
        self.zernike_volume = None

        # Variables Declarations
        self.zernike_coeffs_no_train = None
        self.zernike_coeffs_train = None
        self.zernike_coeffs_no_train2 = None
        # self.height_map = None  # maybe safe memory
        self.channels = None

        self.channels = input_shape[-1]

        if not os.path.exists('zernike_volumes/zernike_volume_%d_n%d.npy' % (self.wave_res[0], self.zernike_terms)):
            if not os.path.exists("zernike_volumes"):
                os.makedirs("zernike_volumes")
            self.zernike_volume = get_zernike_volume(resolution=self.wave_res[0],
                                                     n_terms=self.zernike_terms).astype(np.float32)
            np.save('zernike_volumes/zernike_volume_%d_n%d.npy' % (self.wave_res[0], self.zernike_terms),
                    self.zernike_volume)
        else:
            self.zernike_volume = np.load(
                'zernike_volumes/zernike_volume_%d_n%d.npy' % (self.wave_res[0], self.zernike_terms))

        self.zernike_volume = torch.tensor(self.zernike_volume, dtype=torch.float32, device=self.device)
        num_zernike_coeffs = list(self.zernike_volume.shape)[0]

        zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
        # zernike_inits = np.random.uniform(-1, 1, (num_zernike_coeffs, 1, 1)) / 100
        # zernike_inits[:3] = 0
        if self.optics_cfg == 1:
            zernike_inits[3] = -26
        else:
            zernike_inits[3] = -51  # This sets the defocus value to approximately focus the image for a distance of 1m.

        # zernike_inits[3] = -22
        ##### DEFOCUS #####
        zernike_inits[3] = -22

        self.zernike_coeffs_no_train = nn.Parameter(torch.tensor(zernike_inits[:3, ...], dtype=torch.float32),
                                                    requires_grad=False)
        self.zernike_coeffs_no_train2 = nn.Parameter(torch.tensor(zernike_inits[4:, ...], dtype=torch.float32),
                                                     requires_grad=False)
        self.zernike_coeffs_train = nn.Parameter(torch.tensor(zernike_inits[3, ...], dtype=torch.float32))
        """

        self.zernike_coeffs_no_train = nn.Parameter(torch.tensor(zernike_inits[:3, ...], dtype=torch.float32),
                                                    requires_grad=False)
        self.zernike_coeffs_train = nn.Parameter(torch.tensor(zernike_inits[3:, ...], dtype=torch.float32))"""

        if self.upsample and self.device == torch.device(0):
            print("Images will be upsampled to wave resolution")

        # Training info
        self.training_info = None
        self.gpu_rank = None
        self.experiment = experiment

        mask_0 = np.ones((256, 256, 3))
        c = cv2.circle(img=mask_0, center=[int(256 / 2), int(256 / 2)], radius=32, color=0, thickness=-1,
                       lineType=cv2.FILLED)

        mask_00 = np.zeros((256, 256, 3))
        cc = cv2.circle(img=mask_00, center=[int(256 / 2), int(256 / 2)], radius=32, color=(255, 255, 255),
                        thickness=-1,
                        lineType=cv2.FILLED)

        self.mask_1 = torch.from_numpy(c)
        self.mask_1.unsqueeze(-1).permute(3, 0, 1, 2)
        self.mask_1 = self.mask_1.to(self.device)

        self.mask_2 = torch.from_numpy(cc)
        self.mask_2.unsqueeze(-1).permute(3, 0, 1, 2)
        self.mask_2 = self.mask_2 / self.mask_2.max()
        self.mask_2 = self.mask_2.to(self.device)

    def get_Heith_Map(self):
        ####DEFoCUS####

        zernike_coeffs_concat = torch.cat(
            (self.zernike_coeffs_no_train, self.zernike_coeffs_train.unsqueeze(0), self.zernike_coeffs_no_train2), 0)
        """

        zernike_coeffs_concat = torch.cat((self.zernike_coeffs_no_train, self.zernike_coeffs_train), 0)"""

        height_map = torch.sum(zernike_coeffs_concat * self.zernike_volume, dim=0)
        return height_map.unsqueeze(0)

    def forward(self, input_img, new_zernike=None, prueba=None, psf_lab=None, enfoco=None):
        # ------------------------------------
        # Build height map
        # ------------------------------------

        num_zernike_coeffs = list(self.zernike_volume.shape)[0]

        """
        if new_zernike is None:
            new_zernike = torch.cat((self.zernike_coeffs_no_train, self.zernike_coeffs_train), 0)

        zernike_coeffs_concat = new_zernike


        ####  DEFOCUS  #####
        """

        zernike_coeffs_concat = torch.cat(
            (self.zernike_coeffs_no_train, self.zernike_coeffs_train.unsqueeze(0), self.zernike_coeffs_no_train2), 0)

        # zernike_coeffs_concat[15:] = torch.tensor([0])

        if enfoco is True:
            zernike_coeffs_concat[:] = torch.tensor([0])
            zernike_coeffs_concat[3] = torch.tensor([-22])

        for i in range(num_zernike_coeffs):
            attach_summaries(
                gpu_rank=self.gpu_rank,
                name='zernike_coeff_%d' % i,
                var=zernike_coeffs_concat.flatten()[i],
                train_info=self.training_info,
                experiment=self.experiment,
                only_scalar=True)

        height_map = torch.sum(zernike_coeffs_concat * self.zernike_volume, dim=0)
        height_map = height_map.unsqueeze(0).unsqueeze(-1)

        attach_summaries(gpu_rank=self.gpu_rank, name="Height_map", var=height_map[0],
                         train_info=self.training_info, image=True, log_image=False, experiment=self.experiment)

        element = PhasePlate(wave_lengths=self.wave_lengths,
                             height_map=height_map,
                             refractive_idcs=self.refractive_idcs,
                             height_tolerance=self.height_tolerance)

        # ------------------------------------
        # Get PSF
        # ------------------------------------

        N, M = self.wave_res
        [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)

        x = x / N * self.physical_size
        y = y / M * self.physical_size

        squared_sum = x ** 2 + y ** 2

        wave_nos = 2. * np.pi / self.wave_lengths
        wave_nos = torch.tensor(wave_nos.reshape([1, 1, 1, -1]))
        # [1 / 2, 1 / 1.5, 1 / 1, 1 / 0.5, 1000]
        if self.optics_cfg == 1:
            depth = 1 / 2  # 1.5  # 1/2
        else:
            depth = 1  # 1/2  # self.sensor_distance  # 0: infty, 0.5: 2m, 1: 1m, 1.5: 0.67m, 2: 0.5m (diopters-meters)

        curvature = torch.sqrt(torch.tensor(squared_sum) + torch.tensor(depth, dtype=torch.float64) ** 2)
        curvature = curvature.unsqueeze(0).unsqueeze(-1)

        spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=torch.complex64)

        field = element(spherical_wavefront)
        field = circular_aperture(field)
        sensor_incident_field = propagate_fresnel(field,
                                                  distance=self.sensor_distance,
                                                  sampling_interval=self.sample_interval,
                                                  wave_lengths=self.wave_lengths)
        psf_old = get_intensities(sensor_incident_field)

        ################### FINE TUNING ####################

        if psf_lab is True:
            img = Image.open("../dataset_paula_real/psf_lab.jpg").convert('RGB')
            img = np.array(img)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = cv2.resize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            img = img / img.max()
            psf = torch.FloatTensor(img).to(device)
            psf = psf.unsqueeze(0)
            psf = psf.permute(0, 2, 3, 1)
        else:
            psf = psf_old

        if not self.upsample:
            psf = area_downsampling_tf(psf, self.patch_size)
        psf = torch.div(psf, torch.sum(psf, dim=[1, 2], keepdim=True))
        attach_summaries(gpu_rank=self.gpu_rank, name='PSF', var=psf[0], train_info=self.training_info,
                         image=True, log_image=True, experiment=self.experiment)

        if psf_lab is True:
            psf = torch.where(psf > 3e-5, psf, 0)

        ######################################################################

        ## Prueba 1: loss

        """
        img = psf.squeeze().detach().cpu().numpy()
        zoom_factor = 2
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
        h, w = img.shape[:2]
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

        psf = torch.from_numpy(out)
        psf = psf.unsqueeze(0)"""

        loss = None
        if prueba == "1" or prueba == "3":
            loss = torch.norm(((psf * self.mask_1) - psf))

        if prueba == "2" or prueba == "3":
            psf = psf * self.mask_2

        # Image formation: PSF is convolved with input image
        """c1 = psf.shape[1] // 2
        c2 = psf.shape[2] // 2
        psf = torch.roll(psf, shifts=(c1, c2), dims=(1, 2))"""
        psfs = psf.permute(1, 2, 0, 3)

        # ------------------------------------
        # Get Sensor Image
        # ------------------------------------
        # Upsample input_img to match wave resolution.
        if self.upsample:
            tv_resize = torchvision.transforms.Resize(size=self.wave_res, interpolation=0)
            input_img = tv_resize(input_img)

        sensor_img = img_psf_conv(input_img, psfs)

        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.patch_size)

        noise_sigma = np.random.uniform(low=0.001, high=0.02)

        """
        noisy_img = gaussian_noise(sensor_img,
                                         [self.patch_size, self.patch_size, self.channels],
                                         noise_sigma)
        """
        if self.experiment is not None:
            if sensor_img.shape[0] / self.frames - 1 == 0:
                rimg_summary = 0
            else:
                rimg_summary = np.random.randint(0, sensor_img.shape[0] / self.frames - 1)
            attach_summaries(gpu_rank=self.gpu_rank, name='Sensor_img',
                             var=sensor_img[self.frames * rimg_summary:self.frames * rimg_summary + self.frames],
                             train_info=self.training_info, image=False, log_image=False,
                             experiment=self.experiment, video=True)

        sensor_img = sensor_img / sensor_img.max()
        # sensor_img = torch.div(sensor_img, sensor_img.amax((1, 2, 3))[:, None, None, None])

        # plt.imshow((sensor_img[0] / sensor_img[0].max()).squeeze().permute(1, 2, 0).detach().cpu().numpy()), plt.show()
        # plt.imshow((psf / psf.max()).squeeze().detach().cpu().numpy()), plt.show()

        return sensor_img, psf, zernike_coeffs_concat, loss

    '''
    def extra_repr(self) -> str:
        # (Optional) Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
    '''

    def load_pretrained_from_numpy(self, path):
        weights = np.load(path)['optics_trained_weights']
        state_dict = self.state_dict()
        state_dict['zernike_coeffs_train'].copy_(torch.tensor(weights))

    def load_pretrained_from_warmup(self, path):
        ckpt = torch.load(os.path.expanduser(path), map_location=device)
        state_dict = self.state_dict()
        state_dict['zernike_coeffs_train'].copy_(ckpt['model_state_dict']['optics.zernike_coeffs_train'])
        # state_dict['zernike_coeffs_no_train'][-1].copy_(ckpt['model_state_dict']['optics.zernike_coeffs_train'][0]+2)
        # state_dict['zernike_coeffs_train'].copy_(ckpt['model_state_dict']['optics.zernike_coeffs_train'][1:])


def conv2D(img, kernel):
    img_fft = torch.fft.rfft2(img, dim=(-2, -1))
    kernel_fft = torch.fft.rfft2(kernel, dim=(-2, -1))
    conv = torch.multiply(img_fft, kernel_fft)
    img_conv = torch.fft.irfft2(conv, dim=(-2, -1))
    return img_conv
