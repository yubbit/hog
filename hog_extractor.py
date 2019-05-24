import numpy as np
import cv2 as cv

# Add the following features:
#   - Support gaussian preprocessing
#   - Support different filters
#   - Support signed/unsigned bins
#   - Support different voting methods
# Support custom kernels
# Support more than 2 bin selection for votes

class HOGExtractor:
    def __init__(self, num_bins, cell_sz, block_sz, img_shape=None):
        # Get the bins, their centers, and angle vectors
        bins = np.arange(num_bins+1) / (num_bins)
        bin_ctr = np.pi * (bins + bins[1] / 2)
        bin_ctr = bin_ctr.reshape((1, 1, -1))
        bins = np.pi * bins[:-1]

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

    def _getMaxChannel(self, img_x, img_y, mag, ang):
        # gets the channel with the max magnitude and discards other info
        ch_ix = np.argmax(mag, axis=2)
        y_ix, x_ix = np.indices(mag.shape[0:2])
        res_x = img_x[y_ix, x_ix, ch_ix]
        res_y = img_y[y_ix, x_ix, ch_ix]
        res_mag = mag[y_ix, x_ix, ch_ix]
        res_ang = ang[y_ix, x_ix, ch_ix]

        res_x = np.expand_dims(res_x, -1)
        res_y = np.expand_dims(res_y, -1)
        res_mag = np.expand_dims(res_mag, -1)
        res_ang = np.expand_dims(res_ang, -1)

        return res_x, res_y, res_mag, res_ang

    def _getVoteValues(self, mag, ang):
        h, w, _ = mag.shape
        # get distance from each bin center
        dists = np.absolute(ang - self._bin_ctr)
        dists_ix = np.argsort(dists, axis=2)

        # get the 2nd smallest distance
        y_ix, x_ix = np.indices(mag.shape[0:2])
        mins_ix = dists_ix[:, :, 1]
        mins = dists[y_ix, x_ix, mins_ix]

        # zero out dists not within the top 2
        mins = np.expand_dims(mins, axis=-1)
        mask = (dists <= mins)
        dists = dists * mask

        dists_sum = np.sum(dists, axis=-1)
        dists_sum = np.expand_dims(dists_sum, axis=-1)
        dists = (1 - (dists / dists_sum)) * mask

        votes = dists * mag

        # place votes in pi with votes in 0
        votes[:,:,0] = votes[:,:,-1]
        votes = votes[:,:,:-1]

        return votes

    def _getCellFeatures(self, votes):
        cell_sz = self._cell_sz
        cell_ch = []
        min_y = float('inf')
        min_x = float('inf')

        # collect and sum cells
        for i in range(cell_sz):
            t = votes[i::cell_sz, i::cell_sz, :]
            min_y = min(min_y, t.shape[0])
            min_x = min(min_x, t.shape[1])
            cell_ch.append(t)

        for i in range(cell_sz):
            cell_ch[i] = cell_ch[i][:min_y, :min_x, :]

        cell_vals = np.sum(cell_ch, axis=0)

        return cell_vals

    def getHogFeatures(self, img):
        # convolve the filters and calculate the magnitude and angles
        h, w, ch = img.shape
        img_x = convolve2d(img, self.ker_x, self.ker_x_f)
        img_y = convolve2d(img, self.ker_y, self.ker_y_f)
        mag = np.sqrt(np.square(img_x) + np.square(img_y))
        ang = np.arctan2(img_y, img_x)
        ang = ang + (ang < 0) * np.pi

        _, _, mag, ang = self._getMaxChannel(img_x, img_y, mag, ang)
        votes = self._getVoteValues(mag, ang)
        self._getCellFeatures(votes)

        # basically a vote on bin_ix[i,j,k] will be worth vote[i,j,k]
        # initialize an array of 0s of [width, height, num_bin]
        # use arr[:,:,bin_ix] = vote
        # initialize an array of cells[width/cellsz, height/cellsz, num_bin]
        # iterate and use slicing and skipping to populate this

        cv.imshow('mag', (mag/np.max(mag) * 255).astype(np.uint8))
        cv.imshow('img_x', np.abs(img_x).astype(np.uint8))
        cv.imshow('img_y', np.abs(img_y).astype(np.uint8))
        cv.imshow('img', img)
        cv.imshow('angle', ((ang/(np.pi))*255).astype(np.uint8))
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
img = cv.imread("aya.png")
print(img.shape)
hog = HOGExtractor(9, 8, 16, img.shape)
t = hog.getHogFeatures(img)

