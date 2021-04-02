

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
def segment_error(output, gt):

    # align
    output_align_ID=[]
    for seg_o in output:

        maxL=0
        max_id=-1

        for id in range(len(gt)):
            if seg_intersect(seg_o, gt[id])>maxL:
                max_id=id
                maxL=seg_intersect(seg_o, gt[id])
        output_align_ID.append(max_id)

    # print output_align_ID

    # error
    errL=0
    errN=0
    alpha=20
    print 'alpha: {}'.format(alpha)

    for id in range(len(gt)):
        temp_ids=[i for i in range(len(output_align_ID)) if output_align_ID[i]==id]

        if len(temp_ids)>=1:

            errL+=abs(output[temp_ids[0]][0]-gt[id][0])+abs(output[temp_ids[-1]][1]-gt[id][1])
            errN+=alpha*abs(len(temp_ids)-1)
        else:
            errL+=gt[id][1]-gt[id][0]
            errN+=alpha*abs(len(temp_ids)-1)

    print 'segment length error: {}'.format(errL)
    print 'segment number error: {}'.format(errN)

    return errL+errN


if __name__ == '__main__':

    # print seg_intersect([1,5],[11, 20])
    #l1=[1,2,3,4,5,2]
    #print [i for i in range(len(l1)) if l1[i]==2]
    # gt=[[0, 67], [68, 124], [125, 343], [344, 380], [381, 417]]
    # output=[[0,30],[31,55],[56,75], [76, 150], [151, 240], [241, 300], [301, 366],[367, 417]]
    # print segment_error(output, gt)
    #
    gt=[[0,5,0],[6, 15,1], [16, 33,2], [34,50,3]]
    output=[[0,10,0],[11,22,0],[23,30,1],[31,40,1],[41,45,3],[46,50,2]]

    print segment_error(output, gt)

    pass

