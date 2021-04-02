import numpy as np


def seg_intersect(seg1, seg2):

    if seg2[1]<=seg1[0] or seg2[0]>=seg1[1]:
        return 0
    elif seg2[0]<=seg1[0] and seg2[1]>=seg1[0] and seg2[1]<=seg1[1]:
        return seg2[1]-seg1[0]
    elif seg2[0]>=seg1[0] and seg2[0]<=seg1[1] and seg2[1]>=seg1[1]:
        return seg1[1]-seg2[0]
    else:
        return min(seg1[1]-seg1[0], seg2[1]-seg2[0])


## segment distance
# output: [[seg_x1, seg_y1], ,,, ,[seg_xn, seg_yn]]
# gt: [[gt_x1, gt_y1], ,,, ,[gt_xm, gt_ym]]
# def segment_error(output, gt, alpha):
#     # edit distance
#     # align
#     output_align_ID=[]
#     for seg_o in output:
#
#         maxL=0
#         max_id=-1
#
#         for id in range(len(gt)):
#             if seg_intersect(seg_o, gt[id])>maxL:
#                 max_id=id
#                 maxL=seg_intersect(seg_o, gt[id])
#         output_align_ID.append(max_id)
#
#     # print output_align_ID
#
#     # error
#     errL=0
#     errN=0
#     alpha=alpha
#
#     #print('alpha: {}'.format(alpha))
#
#     for id in range(len(gt)):
#         temp_ids=[i for i in range(len(output_align_ID)) if output_align_ID[i]==id]
#
#         if len(temp_ids)>=1:
#
#             errL+=abs(output[temp_ids[0]][0]-gt[id][0])+abs(output[temp_ids[-1]][1]-gt[id][1])
#             errN+=alpha*abs(len(temp_ids)-1)
#         else:
#             errL+=gt[id][1]-gt[id][0]
#             errN+=alpha*abs(len(temp_ids)-1)
#
#     #print('segment length error: {}'.format(errL))
#     #print('segment number error: {}'.format(errN))
#
#     #return errL+errN, errL, errN
#     print('errL:{} errN:{}'.format(errL, errN))
#     return errL+errN


def segment_error(output, gt, alpha=150):
    # edit distance
    # output: [[seg_x1, seg_y1], ,,, ,[seg_xn, seg_yn]]
    # gt: [[gt_x1, gt_y1], ,,, ,[gt_xm, gt_ym]]
    cps_gt=[0]
    cps_output=[0]

    for seg in gt:
        cps_gt.append(seg[1])
    for seg in output:
        cps_output.append(seg[1]-1)

    cps_output_refined=[]
    err_move=0
    for cp in cps_output:
        idx=np.argmin(np.abs(np.array(cps_gt)-cp))
        cps_output_refined.append(cps_gt[idx])
        err_move+=np.abs(cps_gt[idx]-cp)
    cps_output_refined=np.unique(cps_output_refined)

    err_add=0
    assert len(cps_gt)>=len(cps_output_refined)
    err_add+=len(cps_gt)-len(cps_output_refined)

    #print('err move {}, err add {}'.format(err_move, err_add*alpha))

    return err_move+err_add*alpha


if __name__ == '__main__':

    # print seg_intersect([1,5],[11, 20])
    #l1=[1,2,3,4,5,2]
    #print [i for i in range(len(l1)) if l1[i]==2]
    # gt=[[0, 67], [68, 124], [125, 343], [344, 380], [381, 417]]
    # output=[[0,30],[31,55],[56,75], [76, 150], [151, 240], [241, 300], [301, 366],[367, 417]]
    # print segment_error(output, gt)
    # #
    # gt=[[0,5,0],[6, 15,1], [16, 33,2], [34,50,3]]
    # output=[[0,10,0],[11,22,0],[23,30,1],[31,40,1],[41,45,3],[46,50,2]]
    #
    # print(segment_error(output, gt, alpha))

    pass
    #
    # output=[[0,694],[694,746],[746,1094], [1094,1152],[1152, 1271]]
    # gt=[[0,27,0],[28,172,0],[173,221,0],[222,307,1],[308,466,0],[467,794,1],[795,866,1],[867,1005,2],[1006,1214,2],[1215,1270,0]]
    # segment_error(output, gt, 1)
