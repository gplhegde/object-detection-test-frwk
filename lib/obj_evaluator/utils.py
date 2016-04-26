"""Object detection Accuracy analysis utilities.
"""
def overlap_area(gt_rect, det_rect):
    """Computes overlap area percentage between ground truth box and detected box
    gt_rect  : [xmin, ymain, xmax, ymax]
    det_rect : [xmin, ymain, xmax, ymax]
    """
    
    dx = min(gt_rect[2], det_rect[2]) - max(gt_rect[0] - det_rect[0])
    dy = min(gt_rect[3], det_rect[3]) - max(gt_rect[1] - det_rect[1])

    overlap_area = dx*dy
    gt_area = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
  
    return float(overlap_area) / gt_area
