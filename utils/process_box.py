import numpy as np

def calculate_iou_and_center_distance(A, B):
    # IoU calculation
    x_left = np.maximum(A[:, None, 0], B[:, 0])
    y_top = np.maximum(A[:, None, 1], B[:, 1])
    x_right = np.minimum(A[:, None, 2], B[:, 2])
    y_bottom = np.minimum(A[:, None, 3], B[:, 3])

    intersection_area = np.maximum(x_right - x_left, 0) * np.maximum(y_bottom - y_top, 0)

    area_A = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])

    union_area = (area_A[:, None] + area_B) - intersection_area

    iou = intersection_area / union_area

    # Center distance calculation
    center_A = np.array([(A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2]).T
    center_B = np.array([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]).T

    center_dist = np.linalg.norm(center_A - center_B, axis=1)

    return iou, center_dist


def find_Center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

def find_x_extremes(masks, bboxes, img1_shape, img0_shape):

    gainy = img1_shape[0] / img0_shape[0] 
    gainx = img1_shape[1] / img0_shape[1]
    results = []
    for mask, bbox in zip(masks, bboxes):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate the y-coordinate at the center of the bounding box
        center_y = int((y2+y1)/2)
        
        # Get the row of the mask at center_y
        mask_row = mask[center_y, x1:x2]

        
        if len(mask_row) > 0:
            # Find the leftmost (smallest x) and rightmost (biggest x) non-zero pixels
            non_zero_indices = np.nonzero(mask_row)[:,0]
            if len(non_zero_indices) > 0:
                smallest_x = x1 + non_zero_indices[0]
                biggest_x = x1 + non_zero_indices[-1]
                # Scale back to original image coordinates
                smallest_x_original = int(smallest_x / gainx)
                biggest_x_original = int(biggest_x / gainx)
                center_x_original = int((smallest_x_original + biggest_x_original) / 2)
                center_y_original = int(center_y / gainy)
                results.append([center_x_original, center_y_original])
            else:
                results.append(None)
        else:
            results.append(None)
    return results