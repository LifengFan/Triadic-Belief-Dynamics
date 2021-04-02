import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from data_loader import load_data

# def show_3dposes(ax, pose, color="#3498db", scale=1.0):
#
#     for k in range(26):
#
#         ax.scatter([pose[k][0]], [pose[k][1]], [pose[k][2]], c=color, s=2)
#
#     radius = 2
#
#     fig = plt.figure()  # create a figure object
#     ax = fig.add_subplot(111, projection='3d')  # create an axes object in the figure
#
#     ax.set_xlim3d([-radius, radius])
#     ax.set_ylim3d([-radius, radius])
#     ax.set_zlim3d([-radius, radius])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.get_xaxis().set_ticklabels([])
#     ax.get_yaxis().set_ticklabels([])
#     ax.set_zticklabels([])
#     ax.set_aspect('equal')
#     white = (1.0, 1.0, 1.0, 0.0)
#     ax.w_xaxis.set_pane_color(white)
#     ax.w_yaxis.set_pane_color(white)
#     ax.w_xaxis.line.set_color(white)
#     ax.w_yaxis.line.set_color(white)
#     ax.w_zaxis.line.set_color(white)
#
#     plt.show()
#
#
# def show_3dpointcloud():
#
#     pass






# if __name__=='__main__':
#
#     pose1, pose2, gaze1, gaze2, obj = load_data(1)
#
#
#
#
#     for t in range(len(pose1)):
#         print 'frame id: {}'.format(t)
#
#         try:
#
#             show_3dposes(ax, pose1[t])
#             show_3dposes(ax, pose2[t])
#
#             #plt.clf()
#             plt.close()
#
#         except:
#             pass
#
#








