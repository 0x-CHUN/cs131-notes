import numpy as np
from skimage import feature, data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math


def hog_feature(image, pixel_per_cell=8):
    """
    Compute hog feature for a given image.

    Important: use the hog function provided by skimage to generate both the
    feature vector and the visualization image. **For block normalization, use L1.**

    Args:
        image: an image with object that we want to detect.
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor.

    Returns:
        score: a vector of hog representation.
        hogImage: an image representation of hog provided by skimage.
    """
    ### YOUR CODE HERE
    hogFeature, hogImage = feature.hog(
        image=image,
        pixels_per_cell=(pixel_per_cell, pixel_per_cell),
        visualize=True
    )
    ### END YOUR CODE
    return (hogFeature, hogImage)


def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    """ A sliding window that checks each different location in the image,
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between the hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.

    Hint: use the resize function provided by skimage.

    Args:
        image: an np array of size (h,w).
        base_score: hog representation of the object you want to find, an array of size (m,).
        stepSize: an int of the step size to move the window.
        windowSize: a pair of ints that is the height and width of the window.
    Returns:
        max_score: float of the highest hog score.
        maxr: int of row where the max_score is found (top-left of window).
        maxc: int of column where the max_score is found (top-left of window).
        response_map: an np array of size (h,w).
    """
    # slide a window across the image
    (max_score, maxr, maxc) = (0, 0, 0)
    winH, winW = windowSize
    H, W = image.shape
    pad_image = np.lib.pad(
        image,
        ((winH // 2,
          winH - winH // 2),
         (winW // 2,
          winW - winW // 2)),
        mode='constant')
    response_map = np.zeros((H // stepSize + 1, W // stepSize + 1))
    ### YOUR CODE HERE
    for i in range(0, H + 1, stepSize):
        for j in range(0, W + 1, stepSize):
            hogFeature = feature.hog(
                pad_image[i: i+winH, j:j+winW],
                pixels_per_cell=(pixel_per_cell, pixel_per_cell),
            )
            score = hogFeature.T.dot(base_score)
            response_map[i // stepSize, j // stepSize] = score
            if score > max_score:
                max_score = score
                maxr, maxc = i - winH // 2, j - winW // 2
    ### END YOUR CODE

    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    """
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until either the height or
    width reaches the minimum limit. In the ith iteration,
    the image is resized to scale^i of the original image.

    This function is mostly completed for you -- only a termination
    condition is needed.

    Args:
        image: np array of (h,w), an image to scale.
        scale: float of how much to rescale the image each time.
        minSize: pair of ints showing the minimum height and width.

    Returns:
        images: a list containing pair of
            (the current scale of the image, resized image).
    """
    images = []

    # Yield the original image
    current_scale = 1.0
    images.append((current_scale, image))

    while True:
        # Use "break" to exit this loop if the next image will be smaller than
        # the supplied minimium size
        ### YOUR CODE HERE
        if image.shape[0] < minSize[0] or image.shape[1] < minSize[1]:
            break
        ### END YOUR CODE

        # Compute the new dimensions of the image and resize it
        current_scale *= scale
        image = rescale(image, scale)

        # Yield the next image in the pyramid
        images.append((current_scale, image))

    return images


def pyramid_score(image, base_score, shape, stepSize=20,
                  scale=0.9, pixel_per_cell=8):
    """
    Calculate the maximum score found in the image pyramid using sliding window.

    Args:
        image: np array of (h,w).
        base_score: the hog representation of the object you want to detect.
        shape: shape of window you want to use for the sliding_window.

    Returns:
        max_score: float of the highest hog score.
        maxr: int of row where the max_score is found.
        maxc: int of column where the max_score is found.
        max_scale: float of scale when the max_score is found.
        max_response_map: np array of the response map when max_score is found.
    """
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map = np.zeros(image.shape)
    images = pyramid(image, scale)
    ### YOUR CODE HERE
    for current_scale, image in images:
        score, r, c, response_map = sliding_window(
            image, base_score, stepSize, shape, pixel_per_cell)
        if score > max_score:
            max_score = score
            max_scale = current_scale
            max_response_map = response_map
            maxr, maxc = r, c
    ### END YOUR CODE
    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    """ Calculate the mu and sigma for each part. d is the array
        where each row is the main center (face center) minus the
        part center. Since in our dataset, the face is the full
        image, face center could be computed by finding the center
        of the image. Vector mu is computed by taking an average from
        the rows of d. And sigma is the standard deviation among
        among the rows. Note that the heatmap pixels will be shifted
        by an int, so mu is an int vector.

    Args:
        part_centers: np array of shape (n,2) containing centers
            of one part in each image.
        face_shape: (h,w) that indicates the shape of a face.
    Returns:
        mu: (2,) vector.
        sigma: (2,) vector.

    """
    d = np.zeros((part_centers.shape[0], 2))
    ### YOUR CODE HERE
    d = np.array([face_shape[0] / 2, face_shape[1] / 2]) - part_centers
    mu = d.mean(axis=0).astype(int)
    sigma = d.std(axis=0)
    ### END YOUR CODE
    return mu, sigma


def shift_heatmap(heatmap, mu):
    """First normalize the heatmap to make sure that all the values
        are not larger than 1.
        Then shift the heatmap based on the vector mu.

        Hint: use the interpolation.shift function provided by scipy.ndimage.

        Args:
            heatmap: np array of (h,w).
            mu: vector array of (1,2).
        Returns:
            new_heatmap: np array of (h,w).
    """
    ### YOUR CODE HERE
    heatmap = heatmap / np.max(heatmap)
    row, col = mu
    new_heatmap = np.r_[heatmap[row: , :], heatmap[: row, :]]
    new_heatmap = np.c_[new_heatmap[:, col: ], new_heatmap[:, : col]]
    ### END YOUR CODE
    return new_heatmap


def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    """
    Apply gaussian filter with the given sigmas to the corresponding heatmap.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found.

    Hint: use gaussian function provided by skimage.

    Args:
        image: np array of (h,w).
        sigma: sigma for the gaussian filter.
    Return:
        new_image: an image np array of (h,w) after gaussian convoluted.
    """
    ### YOUR CODE HERE
    new_image = heatmap_face
    for heatmap, sigma in zip(heatmaps, sigmas):
        new_heatmap = gaussian(heatmap, sigma)
        new_image += new_heatmap
    r, c = np.unravel_index(np.argmax(new_image), new_image.shape)
    ### END YOUR CODE
    return heatmap, r, c


def detect_multiple(image, response_map):
    """
    Extra credit
    """
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces
