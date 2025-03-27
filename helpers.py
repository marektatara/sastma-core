import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple


def setup_device() -> torch.device:
    """Return device on which the computations should be performed

    Returns:
        torch.device: device for computation
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device

def save_annotations(anns, path, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    # save the image to path
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def find_bbox(mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return x0, y0, x1, y1

def merge_bboxes(bbox0, bbox1):
    x0 = min(bbox0[0], bbox1[0])
    y0 = min(bbox0[1], bbox1[1])
    x1 = max(bbox0[2], bbox1[2])
    y1 = max(bbox0[3], bbox1[3])
    return x0, y0, x1, y1

def mask_image(image, mask):
    masked_image = image.copy()
    masked_image[~mask] = 0
    return masked_image

def crop_image(image, bbox):
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]

def find_offset(mask):
    y, x = np.where(mask)
    return x.min()

def find_max_nonzero_x(mask0, mask1):
    y, x = np.where(mask0 | mask1)
    return x.max()

def crop_and_align_masks(image0: NDArray, image1: NDArray, mask0: NDArray, mask1: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, int]:

    # Find bounding boxes for masks
    bbox0 = find_bbox(mask0)
    bbox1 = find_bbox(mask1)

    # Merge bboxes to get the common bbox for both masks
    bbox = merge_bboxes(bbox0, bbox1)

    # Remove the background (anything except masks) from the images
    masked_image0 = mask_image(image0, mask0)
    masked_image1 = mask_image(image1, mask1)

    # Crop images
    image0_cropped = crop_image(masked_image0, bbox)
    image1_cropped = crop_image(masked_image1, bbox)

    # Crop masks to fit the images size
    mask0_cropped = crop_image(mask0, bbox)
    mask1_cropped = crop_image(mask1, bbox)

    # Move the left image to the left based on mask and save the offset value. The offset is the first non-zero pixel from the left in the mask.
    offset = find_offset(mask0_cropped)

    print(offset)
    # move the left image to the left

    image0_cropped_moved = np.roll(image0_cropped, -offset, axis=1)

    # move the mask to the left

    mask0_cropped_moved = np.roll(mask0_cropped, -offset, axis=1)

    # crop both images from the right side to the same width, based on both masks

    max_nonzero_x = find_max_nonzero_x(mask0_cropped_moved, mask1_cropped)

    image0_cropped_moved_cropped = image0_cropped_moved[:, :max_nonzero_x]
    image1_cropped_cropped = image1_cropped[:, :max_nonzero_x]

    mask0_cropped_moved_cropped = mask0_cropped_moved[:, :max_nonzero_x]
    mask1_cropped_cropped = mask1_cropped[:, :max_nonzero_x]

    return image0_cropped_moved_cropped, image1_cropped_cropped, mask0_cropped_moved_cropped, mask1_cropped_cropped, offset

def find_left_right_bounds_for_all_heights(mask: NDArray) -> Tuple[NDArray, NDArray]:
    """Computes the leftmost and rightmost non-zero pixel positions for each row in a binary mask.

    Args:
        mask: A 2D binary mask

    Returns:
        left_bounds: An array where each element corresponds to the leftmost non-zero pixel index for each row.
        right_bounds: An array where each element corresponds to the rightmost non-zero pixel index for each row.
    """
    left_bounds = np.zeros(mask.shape[0], dtype=np.int32)
    right_bounds = np.zeros(mask.shape[0], dtype=np.int32)
    for y in range(mask.shape[0]):
        x = np.where(mask[y])[0]
        if len(x) > 0:
            left_bounds[y] = x.min()
            right_bounds[y] = x.max()
    return left_bounds, right_bounds+1
