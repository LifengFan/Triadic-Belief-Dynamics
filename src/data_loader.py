import pickle
import os
from os import path, listdir
import metadata
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

root='/home/lfan/Dropbox/Projects/ECCV20/'

# file=open(path.join(root,'pointclouds/test1/0001.p'),'rb')
# data=cPickle.load(file)
#
# pass

# with open(path.join(root, 'pointclouds/test1/0001.p'),'rb') as filehandle:
#
#     data=pickle.load(filehandle)

# with open(path.join(root, 'post_skeletons/test1/skele1.p'),'rb') as f:
#     skele2 = pickle.load(f)
#
# pass
#----------------------------------------------
# pose

# files=sorted(listdir(path.join(root, 'post_skeletons')))
#
# for i in range(metadata.num_video):
#
#
#     f=metadata.folder_name_map[str(i+1)]
#
#     pose1=pickle.load(open(path.join(root, 'post_skeletons',f,'skele1.p'), 'rb'))
#     pose2=pickle.load(open(path.join(root, 'post_skeletons',f,'skele2.p'),'rb'))
#
#     pass
#-----------------------------------------------
# gaze

# data=pickle.load(open(path.join(root,'merge2gaze','test2','target.p'),'rb'))
# data=joblib.load(open(path.join(root,'merge2gaze','test1','target.p'),'rb'))

# for i in range(metadata.num_video):
#
#     f=metadata.folder_name_map[str(i+1)]
#
#     # gaze=joblib.load(open(path.join(root, 'merge2gaze', f, 'target.p'), 'rb'))
#     gaze = joblib.load(open(path.join(root, 'tracker_gaze', f, 'output.p'), 'rb'))
#
#     pass
#------------------------------------------------
# object
#------------------------------------------------
# annotation
#-------------------------------------------------

# for vid in range(metadata.num_video):

def load_data(folder_name):

    #f=metadata.folder_name_map[str(vid)]

    print('loading data: {}'.format(folder_name))

    pose1=joblib.load(open(path.join(root, 'data', 'pose', 'post_skeletons', folder_name, 'skele1.p'), 'rb'))
    pose2=joblib.load(open(path.join(root, 'data', 'pose', 'post_skeletons', folder_name, 'skele2.p'), 'rb'))

    gaze1=joblib.load(open(path.join(root, 'data', 'gaze', 'merge2gaze_tracker', folder_name, 'target.p'), 'rb'))
    gaze2 = joblib.load(open(path.join(root, 'data', 'gaze', 'merge2gaze_battery', folder_name, 'target.p'), 'rb'))

    obj_per_frame=joblib.load(open(path.join(root, 'data', 'object', 'track_cate_with_frame', folder_name, folder_name + '.p'), 'rb'))
    obj = joblib.load(open(path.join(root, 'data', 'object', 'track_cate', folder_name, folder_name + '.p'), 'rb'))

    T=len(pose1)
    assert len(pose2)==T
    assert len(gaze1)==T
    assert len(gaze2)==T

    pointclouds={}
    for t in range(T):
        pointclouds[t]=joblib.load(open(path.join(root, 'data', 'pointclouds', folder_name, str(t).zfill(4) + '.p'), 'rb'))


    return pose1, pose2, gaze1, gaze2, obj_per_frame, obj, pointclouds, T

# def k_means_cluster():
#     # use some measure to calculate the similarity
#     # need to set a cluster number
#
#     pass


# def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
#   """
#   Visualize a 3d skeleton
#   Args
#     channels: 96x1 vector. The pose to plot.
#     ax: matplotlib 3d axis to draw on
#     lcolor: color for left part of the body
#     rcolor: color for right part of the body
#     add_labels: whether to add coordinate labels
#   Returns
#     Nothing. Draws on ax.
#   """
#
#   #assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
#   #vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )
#
#   # I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
#   # J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
#   # LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
#
#   # Make connection matrix
#   for i in np.arange( len(I) ):
#     x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
#     ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
#
#   RADIUS = 750 # space around the subject
#   xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]



class visualize_data():

    def __init__(self,folder_name):

        self.folder_name=folder_name
        print('visualize: {}'.format(self.folder_name))

        pose1, pose2, gaze1, gaze2, obj_per_frame, obj, pointclouds, T = load_data(self.folder_name)

        self.pose1=pose1
        self.pose2=pose2
        self.gaze1=gaze1
        self.gaze2=gaze2
        self.obj_per_frame=obj_per_frame
        self.obj=obj
        self.pointclouds=pointclouds
        self.T=T

        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)

        self.radius = 4
        self.ax.set_xlim3d([-self.radius, self.radius])
        self.ax.set_ylim3d([-self.radius, self.radius])
        self.ax.set_zlim3d([-self.radius, self.radius])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(self.folder_name)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.get_xaxis().set_ticklabels([])
        self.ax.get_yaxis().set_ticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_aspect('equal')
        white = (1.0, 1.0, 1.0, 0.0)
        self.ax.w_xaxis.set_pane_color(white)
        self.ax.w_yaxis.set_pane_color(white)
        self.ax.w_xaxis.line.set_color(white)
        self.ax.w_yaxis.line.set_color(white)
        self.ax.w_zaxis.line.set_color(white)
        self.ax.view_init(azim=-90, elev=-55)


    def init(self):

        # pose
        p1_xs=[self.pose1[0][i][0] for i in range(26)]
        p1_ys=[self.pose1[0][i][1] for i in range(26)]
        p1_zs=[self.pose1[0][i][2] for i in range(26)]

        self.p1=self.ax.scatter3D(p1_xs, p1_ys, p1_zs, c="#3498db", s=2)

        p2_xs = [self.pose2[0][i][0] for i in range(26)]
        p2_ys = [self.pose2[0][i][1] for i in range(26)]
        p2_zs = [self.pose2[0][i][2] for i in range(26)]

        self.p2=self.ax.scatter3D(p2_xs, p2_ys, p2_zs, c="#e74c3c", s=2)

        # gaze
        # need to normalize
        self.g1=self.ax.plot([], [], [], c='#7b4173')[0]
        self.g2=self.ax.plot([], [], [], c='#4daf4a')[0]


        try:
            g1_xs=[self.gaze1[0][0][0], self.gaze1[0][1][0]]
            g1_ys=[self.gaze1[0][0][1], self.gaze1[0][1][1]]
            g1_zs=[self.gaze1[0][0][2], self.gaze1[0][1][2]]

            self.g1=self.ax.plot(g1_xs, g1_ys, g1_zs, c='#7b4173')[0]

            g2_xs = [self.gaze2[0][0][0], self.gaze2[0][1][0]]
            g2_ys = [self.gaze2[0][0][1], self.gaze2[0][1][1]]
            g2_zs = [self.gaze2[0][0][2], self.gaze2[0][1][2]]

            self.g2 = self.ax.plot(g2_xs, g2_ys, g2_zs, c='#4daf4a')[0]

        except:

            pass

        # object
        for i in range(len(self.pointclouds[0])):
            print(self.pointclouds[0][i][0])

            for j in range(len(self.pointclouds[0][i][1])):

                point_xs=[self.pointclouds[0][i][1][j][k][0] for k in range(len(self.pointclouds[0][i][1][j]))]
                point_ys = [self.pointclouds[0][i][1][j][k][1] for k in range(len(self.pointclouds[0][i][1][j]))]
                point_zs = [self.pointclouds[0][i][1][j][k][2] for k in range(len(self.pointclouds[0][i][1][j]))]

                self.ax.scatter3D(point_xs,point_ys,point_zs, c="#983334", s=0.5)

        pass



        return self.p1, self.p2, self.g1, self.g2

    def update(self, ind):

        p1_xs=[self.pose1[ind][i][0] for i in range(26)]
        p1_ys=[self.pose1[ind][i][1] for i in range(26)]
        p1_zs=[self.pose1[ind][i][2] for i in range(26)]

        self.p1._offsets3d=(p1_xs, p1_ys, p1_zs)

        p2_xs=[self.pose2[ind][i][0] for i in range(26)]
        p2_ys=[self.pose2[ind][i][1] for i in range(26)]
        p2_zs=[self.pose2[ind][i][2] for i in range(26)]


        self.p2._offsets3d=(p2_xs, p2_ys, p2_zs)

        try:
            g1_xs = [self.gaze1[ind][0][0], self.gaze1[ind][1][0]]
            g1_ys = [self.gaze1[ind][0][1], self.gaze1[ind][1][1]]
            g1_zs = [self.gaze1[ind][0][2], self.gaze1[ind][1][2]]

            self.g1.set_data(g1_xs, g1_ys)
            self.g1.set_3d_properties(g1_zs)

            g2_xs = [self.gaze2[ind][0][0], self.gaze2[ind][1][0]]
            g2_ys = [self.gaze2[ind][0][1], self.gaze2[ind][1][1]]
            g2_zs = [self.gaze2[ind][0][2], self.gaze2[ind][1][2]]

            self.g2.set_data(g2_xs, g2_ys)
            self.g2.set_3d_properties(g2_zs)


        except:
            pass

        return self.p1, self.p2, self.g1, self.g2


    def animate(self):

        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.T, blit=False, init_func=self.init)

        plt.show()


if __name__ == '__main__':

    # for key in metadata.annot_map.keys():
    #
    #     vis_data=visualize_data(key)
    #     vis_data.animate()
    #
    #     pass

    pose1 = joblib.load(open('segments.p', 'rb'))

    pass






