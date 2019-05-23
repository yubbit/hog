import numpy as np
import cv2 as cv

# Add the following features:
#   - Support gaussian preprocessing
#   - Support different filters
#   - Support signed/unsigned bins
#   - Support different voting methods
# Memoise the Fourier representations of the two filters

class HOGExtractor:

    def __init__(self, num_bins, cell_sz, block_sz, img_shape=None):
        # Get the bins, their centers, and angle vectors
        bins = np.arange(num_bins+1) / (num_bins)
        bins = bins[:-1]
        bin_ctr = np.pi * (bins[:-1] + bins[1] / 2)

        # Store the kernels and their Fourier representations
        ker_x = np.array(((-1, 0, 1),))
        ker_y = ker_x.T

        # Pre-compute kernel Fourier representations if possible
        if img_shape is not None:
            m, n = img_shape[0:2]
            k, l = ker_x.shape[0:2]
            ker_x_f = np.fft.fft2(ker_x, (m+k-1, n+l-1), (0, 1))

            k, l = ker_y.shape[0:2]
            ker_y_f = np.fft.fft2(ker_y, (m+k-1, n+l-1), (0, 1))

        else:
            ker_x_f = None
            ker_y_f = None

        self._num_bins = num_bins
        self._cell_sz = cell_sz
        self._block_sz = block_sz

        self._bins = bins
        self._bin_ctr = bin_ctr

        self.ker_x = ker_x
        self.ker_y = ker_y
        self.ker_x_f = ker_x_f
        self.ker_y_f = ker_y_f

    def getHogFeatures(self, img):
        # convolve the filters and calculate the magnitude
        img_x = convolve2d(img, self.ker_x, self.ker_x_f)
        img_y = convolve2d(img, self.ker_y, self.ker_y_f)
        mag = np.sqrt(np.square(img_x) + np.square(img_y))

        # get the channel with the highest magnitude per pixel
        mag_amax = np.argmax(mag, axis=2)
        x_ix, y_ix = np.indices(img.shape[0:2])

        # retain only the information on the highest magnitude pixel
        mag = mag[x_ix, y_ix, mag_amax]
        img_x = img_x[x_ix, y_ix, mag_amax]
        img_y = img_y[x_ix, y_ix, mag_amax]

        # get the angle at each point for binning
        # flip negative angles so theyre on the range 0 to pi/2
        angle = np.arctan2(img_y, img_x)
        angle = angle + (angle < 0) * np.pi / 2

        cv.imshow('mag', (mag/np.max(mag) * 255).astype(np.uint8))
        cv.imshow('img_x', np.abs(img_x).astype(np.uint8))
        cv.imshow('img_y', np.abs(img_y).astype(np.uint8))
        cv.imshow('img', img)
        cv.imshow('angle', ((angle/(np.pi))*200).astype(np.uint8))
        cv.waitKey(0)

def convolve2d_gray(img, ker, ker_f=None):
    m, n = img.shape[0:2]
    k, l = ker.shape[0:2]

    # Zero-padded convolution
    img_f = np.fft.fft2(img, (m+k-1, n+l-1), (0, 1))
    # If kernel transform pre-computed, use that
    if ker_f is None:
        ker_f = np.fft.fft2(ker, (m+k-1, n+l-1), (0, 1))

    res_f = img_f * ker_f
    res = np.fft.ifft2(res_f, axes=(0, 1)).real

    # Roll and slice to achieve SAME padding
    res = np.roll(res, ((-k//2)+1, (-l//2)+1), (0, 1))
    res = res[0:m, 0:n]

    return res

def convolve2d(img, ker, ker_f=None):
    # Helper function for 2D convolution over multiple channels
    if len(img.shape) == 3:
        arrs = list()
        for i in range(img.shape[2]):
            arrs.append(convolve2d_gray(img[:,:,i], ker, ker_f))
        res = np.stack(arrs, axis=-1)
    else:
        res = convolve2d_gray(img, ker, ker_f)
    return res

img = cv.imread("giraffe.jpg")
hog = HOGExtractor(9, 8, 16, img.shape)
t = hog.getHogFeatures(img)

