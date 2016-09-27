import cv2
import numpy as np
from matplotlib import pyplot as plt

class Saliency:

    def __init__(self, img, use_numpy_fft=True, gauss_kernel=(5, 5)):
        self.use_numpy_fft = use_numpy_fft
        self.gauss_kernel = gauss_kernel
        self.frame_orig = img

        self.small_shape = (64, 64)
        self.frame_small = cv2.resize(img, self.small_shape[1::-1])

        # whether we do need to do the math (True) or it has already been done (False)
        self.need_saliency_map = True

    def plot_magnitude(self):
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols-cols, 0, nrows-rows, cv2.BORDER_CONSTANT, value=0)

        img_dft = np.fft.fft2(frame)
        spectrum = np.log10(np.real(np.abs(np.fft.fftshift(img_dft))))
        #magn = np.abs(img_dft)
        #log_magn = np.log10(magn)
        #spectrum = fft.fftshift(log_magn)
        #return frame
        return 255*spectrum/np.max(spectrum)

    def get_saliency_map(self):

        if self.need_saliency_map:
            num_channels = 1
            if len(self.frame_orig.shape) == 2:
                sal = self._get_channel_sal_map(self.frame_small)
            else:
                sal = np.zeros_like(self.frame_small).astype(np.float32)
                for c in xrange(self.frame_small.shape[2]):
                    small = self.frame_small[:, :, c]
                    sal[:, :, c] = self._get_channel_sal_magn(small)

                sal = np.mean(sal, 2)

            if self.gauss_kernel is not None:
                sal = cv2.GaussianBlur(sal, self.gauss_kernel, sigmaX=8, sigmaY=0)

            sal = sal**2
            sal = np.float32(sal)/np.max(sal)

            sal = cv2.resize(sal, self.frame_orig.shape[1::-1])

            self.saliencyMap = sal
            self.need_saliency_map = False

        return self.saliencyMap

    def _get_channel_sal_magn(self, channel):

        if self.use_numpy_fft:
            img_dft = np.fft.fft2(channel)
            magnitude, angle = cv2.cartToPolar(np.real(img_dft),
                                               np.imag(img_dft))
        else:
            img_dft = cv2.dft(np.float32(channel),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude, angle = cv2.cartToPolar(img_dft[:, :, 0],
                                               img_dft[:, :, 1])
        log_ampl = np.log10(magnitude.clip(min=1e-9))

        log_ampl_blur = cv2.blur(log_ampl, (3, 3))

        residual = np.exp(log_ampl - log_ampl_blur)

        if self.use_numpy_fft:
            real_part, imag_part = cv2.polarToCart(residual, angle)
            img_combined = np.fft.ifft2(real_part + 1j*imag_part)
            magnitude, _ = cv2.cartToPolar(np.real(img_combined),
                                                   np.imag(img_combined))
        else:
            img_dft[:, :, 0], img_dft[:, :, 1] = cv2.polarToCart(residual,
                                                                 angle)
            img_combined = cv2.idft(img_dft)
            magnitude, _ = cv2.cartToPolar(img_combined[:, :, 0],
                                           img_combined[:, :, 1])


        return magnitude




    def get_proto_objects(self, use_otsu=False):

        saliency = self.get_saliency_map()

        if use_otsu:
            _, img_objects = cv2.threshold(np.uint8(saliency*255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            thresh = np.mean(saliency)*255*3
            _, img_objects = cv2.threshold(np.uint8(saliency*255), thresh, 255, cv2.THRESH_BINARY)

        return img_objects


    def plot_power_density(self):
        pass

    def plot_power_spectrum(self):
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols - cols, 0, nrows - rows, cv2.BORDER_CONSTANT, value=0)

        if self.use_numpy_fft:
            img_dft = np.fft.fft2(frame)
            spectrum = np.log10(np.real(np.abs(img_dft))**2)
        else:
            img_dft = cv2.dft(np.float32(frame),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            spectrum = np.log10(img_dft[:, : , 0]**2 + img_dft[:, :, 1]**2)

        L = max(frame.shape)
        freqs = np.fft.fftfreq(L)[:L/2]
        dists = np.sqrt(
            np.fft.fftfreq(frame.shape[0])[:, np.newaxis]**2 +
            np.fft.fftfreq(frame.shape[1])**2)
        dcount = np.histogram(dists.ravel(), bins=freqs)[0]
        histo, bins = np.histogram(dists.ravel(), bins=freqs, weights=spectrum.ravel())
        centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(centers, histo/dcount)
        plt.xlabel('frequency')
        plt.ylabel('log-spectrum')
        plt.show()
