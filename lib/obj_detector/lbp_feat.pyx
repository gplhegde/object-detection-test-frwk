import numpy as np
cimport numpy as np

cdef inline np.int32_t _get_cell_sum(np.ndarray[np.int32_t, ndim=2] ii_img, int x,int y, int w, int h):
    cdef int left_pt, top_pt
    left_pt = x-1
    top_pt = y-1
    # a --> top left corner of ii image 
    if (left_pt < 0 or top_pt < 0):
        a = 0
    else:
        a = ii_img[top_pt, left_pt]
    # top right corner
    if(top_pt < 0):
        b = 0
    else:
        b = ii_img[top_pt, left_pt + w]
    # bottom left corner
    if(left_pt < 0):
        d = 0
    else:
        d = ii_img[top_pt+h, left_pt]
    # bottom right point
    c = ii_img[top_pt+h, left_pt+w]
    # cell sum
    return (a+c) - (b+d)

def lbp_feat(np.ndarray[np.int32_t, ndim=2] ii_img, int ft_x, int ft_y, int ft_w, int ft_h, int win_pos_x, int win_pos_y):
    """Returns lbp code given the feature parameters (x, y, w, h) in the image and 
    integral image.
    Output: lab code (np.uint8)
    """
    cdef np.uint8_t lbp_code
    cdef int feat_x, feat_y, cell_width, cell_height, n, cell_sum, cent_x, cent_y
    # move the co-ordinate from window's top left corner to feature block's top left corner
    feat_x = win_pos_x + ft_x
    feat_y = win_pos_y + ft_y
    cell_width = ft_w
    cell_height = ft_h
    # cell positions in terms of multiple of cell height and width in the LBP order. (x, y) format
    #cell_pos = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    cdef int cell_pos_x[8], cell_pos_y[8]
    cell_pos_x[:] = [0, 1, 2, 2, 2, 1, 0, 0]
    cell_pos_y[:] = [0, 0, 0, 1, 2, 2, 2, 1]
    # sum of center cell
    cent_x = feat_x + cell_width
    cent_y = feat_y + cell_height
    cent_sum = _get_cell_sum(ii_img, cent_x, cent_y, cell_width, cell_height)
    # compute sum of all 8 cells and generate LBP code
    lbp_code = 0
    for n in range(8):
        cell_x = feat_x + cell_pos_x[n] * cell_width
        cell_y = feat_y + cell_pos_y[n] * cell_height
        cell_sum = _get_cell_sum(ii_img, cell_x, cell_y, cell_width, cell_height)
        if(cell_sum > cent_sum):
            lbp_code += (2**(7-n))

    return lbp_code
