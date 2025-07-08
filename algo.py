import cv2
import numpy as np
from helpers import find_left_right_bounds_for_all_heights
from tqdm import tqdm
from numpy.typing import NDArray

def calculate_sad_values(image0: NDArray, image1: NDArray, mask0: NDArray, mask1: NDArray, window_size: int, maxy: int=None) -> NDArray:
    """Calculate the Sum of Absolute Differences (SAD) values between two images with a specified window size.

    Args:
        image0: The left input image (2D array).
        image1: The right input image (2D array).
        mask0: Binary mask for the left image, defining valid regions.
        mask1: Binary mask for the right image, defining valid regions.
        window_size: The size of the square window used for SAD computation. Must be an odd number.
        maxy (optional): The maximum number of y-coordinates to process. If None, process all rows.
    Returns:
        sads: A 3D array of SAD values with shape (image0_height, image0_width, image1_width). Values are set to inf where the mask is not present.
    """

    # the images are padded to handle the window size
    pad = window_size // 2
    image0_padded = cv2.copyMakeBorder(image0, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    image1_padded = cv2.copyMakeBorder(image1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    left_bounds0, right_bounds0 = find_left_right_bounds_for_all_heights(mask0)
    left_bounds1, right_bounds1 = find_left_right_bounds_for_all_heights(mask1)
    sad_values = np.zeros((image0.shape[0], image0.shape[1], image1.shape[1]), dtype=np.float32)

    # set values to infinity where the mask is not present
    sad_values[:] = np.inf

    # helper to iterate over the pixels within the masks
    x0_ranges = [range(left_bounds0[y], right_bounds0[y]) for y in range(image0.shape[0])]
    x1_ranges = [range(left_bounds1[y], right_bounds1[y]) for y in range(image0.shape[0])]

    # now, for each row compare pairs of blocks and calculate the SAD for each pair possible
    for y, x_range, x1_range in tqdm(zip(range(image0.shape[0] if maxy is None else maxy+1), x0_ranges, x1_ranges), total=image0.shape[0] if maxy is None else maxy+1):
        left_blocks = np.array([image0_padded[y:y+2*pad+1, x:x+2*pad+1] for x in x_range])
        right_blocks = np.array([image1_padded[y:y+2*pad+1, x1:x1+2*pad+1] for x1 in x1_range])
        if len(left_blocks) > 0 and len(right_blocks) > 0:
            diffs = left_blocks[:, np.newaxis] - right_blocks
            sad_values[y, x_range, x1_range[0]:x1_range[-1]+1] = np.abs(diffs).sum(axis=(2, 3, 4))
    # normalize the SAD values by the window size
    sad_values = sad_values//window_size**2

    sad_values[np.isnan(sad_values)] = np.inf
    return sad_values

def find_confidence(sad, left_index):
    index_sad = sad[left_index, :]
    min_sad_value = index_sad.min()
    min_sad_index = index_sad.argmin()
    angles = np.abs(np.arctan2(np.array(range(0, sad.shape[1])) - min_sad_index, min_sad_value - index_sad))
    # find second minimal value
    second_min_sad_value = angles[angles != 0].min()
    return second_min_sad_value

