#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:45:03 2018

@author: uzielr
"""

"""
************************************************************************************************************************************************************

                                                                            Imports

************************************************************************************************************************************************************
"""
import cProfile
import torch
from scipy.special import gamma, factorial,multigammaln
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
import gc
import Global
import Sample_Functions as my_sample
import Help_Functions as my_help
import Plot_Functions as my_plot
import Conn_Functions as my_connectivity
from torch.autograd import Variable
from colorama import Fore, Back, Style
import numpy as np
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')
import cv2
from numpy.linalg import inv
from numpy.linalg import det
import warnings
from copy import deepcopy
#import cuda_sp
from torch.distributions import StudentT

warnings.filterwarnings("ignore")

""" "#000000",
"""

"""
************************************************************************************************************************************************************

                                                                        Global Params

************************************************************************************************************************************************************
"""

"""
************************************************************************************************************************************************************

                                                                            Start

************************************************************************************************************************************************************
"""


def SuperPixelsV2():
    """Generating SuperPixels over location , intensity and optical flow
    **Parameters**:
     -
    **Returns**:
     - Figure with SupeerPixels
    """

    Y = np.zeros((Global.N, Global.C_D))
    H = np.zeros((Global.N, Global.C_D, Global.D))
    [Ix, Iy, It] = Compute_Gradient(Global.frame0, Global.frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It




    XData = my_help.Create_DataMatrix(Global.frame0)
    KmeansSP2(XData, H, Y)


def SuperPixelsV2TF():
    """Generating SuperPixels over location , intensity and optical flow
    **Parameters**:
     -
    **Returns**:
     - Figure with SupeerPixels
    """
    Y = np.zeros((Global.N, Global.C_D))
    H = np.zeros((Global.N, Global.C_D, Global.D))
    [Ix, Iy, It] = Compute_Gradient(Global.frame0, Global.frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It

    # Y0 = np.zeros((Global.N, Global.C_D))
    # H0 = np.zeros((Global.N, Global.C_D, Global.D))
    # frame1 = cv2.imread(Global.IMAGE2)
    # [Ix, Iy, It] = Compute_Gradient(Global.frame0, frame1)
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    #
    # H0[:, :, 0] = Ix
    # H0[:, :, 1] = Iy
    # Y0[:, :] = -It
    #
    # scale = 0.75
    # frame0_1 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # frame1_1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # H1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D, Global.D))
    # Y1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D))
    # [Ix, Iy, It] = Compute_Gradient(frame0_1, frame1_1)
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H1[:, :, 0] = Ix
    # H1[:, :, 1] = Iy
    # Y1[:, :] = -It * 0.75
    #
    # H = np.append(H0, H1, axis=0)
    # Y = np.append(Y0, Y1, axis=0)
    #
    # scale = 0.5
    # frame0_2 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # frame1_2 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # H2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D, Global.D))
    # Y2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D))
    #
    # [Ix, Iy, It] = Compute_Gradient(frame0_2, frame1_2)
    #
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H2[:, :, 0] = Ix
    # H2[:, :, 1] = Iy
    # Y2[:, :] = -It * 0.5
    #
    # H = np.append(H, H2, axis=0)
    # Y = np.append(Y, Y2, axis=0)
    # scale = 0.25
    # frame0_3 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # frame1_3 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # H3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D, Global.D))
    # Y3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D))
    #
    # [Ix, Iy, It] = Compute_Gradient(frame0_3, frame1_3)
    #
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H3[:, :, 0] = Ix
    # H3[:, :, 1] = Iy
    # Y3[:, :] = -It * 0.25
    #
    # H = np.append(H, H3, axis=0)
    # Y = np.append(Y, Y3, axis=0)

    XData = my_help.Create_DataMatrix(Global.frame0)
    KmeansSP2TF(XData, H, Y)


def SuperPixelsCuda():
    """Generating SuperPixels over location , intensity and optical flow
    **Parameters**:
     -
    **Returns**:
     - Figure with SupeerPixels
    """

    Y = np.zeros((Global.N, Global.C_D))
    H = np.zeros((Global.N, Global.C_D, Global.D))
    [Ix, Iy, It] = Compute_Gradient(Global.frame0, Global.frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It

    r_ik, pi = InitKmeansSP()
    maxIt = 10000
    it = 0
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = Z / (Zmax)
    # Z[Z==0]=np.nan
    #    Z=np.prod(Z,axis=1)
    threshold = np.percentile(Z,0.0000001)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0

    XData = my_help.Create_DataMatrix(Global.frame0)
    XData = XData.astype(dtype=np.float32)
    H = H.astype(dtype=np.float32)
    Y = Y.astype(dtype=np.float32)
    KmeansSPCuda(XData, H, Y)
    a = 3


def SuperPixelsV1():
    """Generating SuperPixels over location , intensity
    **Parameters**:
     -
    **Returns**:
     - Figure with superpixels
    """

    X = my_help.Create_DataMatrix(Global.frame0)
    KmeansSP(X)


def SuperPixelsV1TF():
    """Generating SuperPixels over location , intensity
    **Parameters**:
     -
    **Returns**:
     - Figure with superpixels
    """

    X = my_help.Create_DataMatrix(Global.frame0)
    KmeansSPTF(X)


def PyramidFlow():
    """Clusters flow using soft K-means.
       Using 3 resolutions of images : 1,0.75.0.5
    **Parameters**:
     -
    **Returns**:
     - Figure with the clusterd points
    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.
    .. note:: Using inter-linear interpolation in order to downsample
    .. note:: Weight of each resolution defined in k-means
    todo:: Change downsample size/weights/interpolation

    """
    Y0 = np.zeros((Global.N, Global.C_D))
    H0 = np.zeros((Global.N, Global.C_D, Global.D))
    frame1 = cv2.imread(Global.IMAGE2)
    [Ix, Iy, It] = Compute_Gradient(Global.frame0, frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))

    H0[:, :, 0] = Ix
    H0[:, :, 1] = Iy
    Y0[:, :] = -It

    scale = 0.75
    frame0_1 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame1_1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D, Global.D))
    Y1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D))
    [Ix, Iy, It] = Compute_Gradient(frame0_1, frame1_1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H1[:, :, 0] = Ix
    H1[:, :, 1] = Iy
    Y1[:, :] = -It * 0.75

    H = np.append(H0, H1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    scale = 0.5
    frame0_2 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame1_2 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    H2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D, Global.D))
    Y2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D))

    [Ix, Iy, It] = Compute_Gradient(frame0_2, frame1_2)

    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H2[:, :, 0] = Ix
    H2[:, :, 1] = Iy
    Y2[:, :] = -It * 0.5

    H = np.append(H, H2, axis=0)
    Y = np.append(Y, Y2, axis=0)
    scale = 0.25
    frame0_3 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame1_3 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    H3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D, Global.D))
    Y3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D))

    [Ix, Iy, It] = Compute_Gradient(frame0_3, frame1_3)

    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H3[:, :, 0] = Ix
    H3[:, :, 1] = Iy
    Y3[:, :] = -It * 0.25

    H = np.append(H, H3, axis=0)
    Y = np.append(Y, Y3, axis=0)

    mu_max, clusters, Cmax = Kmeans(H, Y)
    return


def ArttificalFlowPyramid():
    """Creates articifal flow and clusters it using soft K-means.
       Using 3 resolutions of images : 1,0.75.0.5
    **Parameters**:
     -
    **Returns**:
     - Figure with the clusterd points
    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.
    .. note:: Using inter-linear interpolation in order to downsample
    .. note:: Weight of each resolution defined in k-means
    .. note:: Can use real flow by adding the comment line in the function
    todo:: Change downsample size/weights/interpolation

    """
    Y0 = np.zeros((Global.N, Global.C_D))
    H0 = np.zeros((Global.N, Global.C_D, Global.D))
    Xk, X, U, V, MU, SIGMA, C, framePoints = my_sample.SampleXK()
    print(Back.BLUE)
    print("Number of clusters: ", Global.K)
    for i in range(0, Global.K):
        print("Real Mu[", i, "] :", MU[i], "Real C[", i, "]: ", C[i])
    print(Style.RESET_ALL)

    maxXY = np.amax([np.amax(X[0, :]), np.amax(X[1, :])])
    minXY = np.amin([np.amin(X[0, :]), np.amin(X[1, :])])

    # Set Axes
    fig, ax = plt.subplots(3, 2)
    for (m, n), subplot in np.ndenumerate(ax):
        subplot.set_xlim(minXY, maxXY)
        subplot.set_ylim(minXY, maxXY)
        plt.axis("equal")

    yy, xx = np.mgrid[:Global.HEIGHT, :Global.WIDTH]
    xx = xx - U
    yy = yy - V
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    plt.clf()

    """
    ax = plt.subplot(321)
    ax.set_title("Frame 0")
    plt.imshow(cv2.cvtColor(Global.frame0, cv2.COLOR_BGR2RGB))
    """

    frame1 = cv2.remap(Global.frame0, xx, yy, cv2.INTER_CUBIC)

    frame1 = cv2.imread(Global.IMAGE2)  # Add this line in order to use 2 frames

    """
    ax = plt.subplot(322)
    ax.set_title("Frame 1")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

    ax = plt.subplot(323)
    ax.set_title("Real Points and contour")

    for i in range(0, Global.K):
        color = Global.colors[i + 1]
        my_plot.PlotContour(MU[i], SIGMA[i], Xk[i], color, 'Original-' + str(i))
        my_plot.PlotPoints(Xk[i], color, 'X' + str(i) + '- Data')

    ax = plt.subplot(324)
    framePoints = framePoints.astype('uint8')
    plt.imshow(framePoints)
    """

    [Ix, Iy, It] = Compute_Gradient(Global.frame0, frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))

    H0[:, :, 0] = Ix
    H0[:, :, 1] = Iy
    Y0[:, :] = -It

    scale = 0.5
    frame0_1 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame1_1 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    H1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D, Global.D))
    Y1 = np.zeros((frame0_1.shape[0] * frame0_1.shape[1], Global.C_D))
    [Ix, Iy, It] = Compute_Gradient(frame0_1, frame1_1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))
    H1[:, :, 0] = Ix
    H1[:, :, 1] = Iy
    Y1[:, :] = -It * 0.5

    H1 = H1.reshape((frame0_1.shape[0], frame0_1.shape[1], 3, 2))
    Y1 = Y1.reshape((frame0_1.shape[0], frame0_1.shape[1], 3))
    H1 = H1.repeat(2, axis=0).repeat(2, axis=1)
    Y1 = Y1.repeat(2, axis=0).repeat(2, axis=1)
    H1 = H1.reshape((-1, 3, 2))
    Y1 = Y1.reshape((-1, 3))
    H = np.append(H0, H1, axis=1)
    Y = np.append(Y0, Y1, axis=1)

    # scale = 0.25
    # frame0_2 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # frame1_2 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # H2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D, Global.D))
    # Y2 = np.zeros((frame0_2.shape[0] * frame1_2.shape[1], Global.C_D))
    # [Ix, Iy, It] = Compute_Gradient(frame0_2, frame1_2)
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H2[:, :, 0] = Ix
    # H2[:, :, 1] = Iy
    # Y2[:, :] = -It * 0.25
    #
    # H2 = H2.reshape((frame0_2.shape[0], frame0_2.shape[1], 3, 2))
    # Y2 = Y2.reshape((frame0_2.shape[0], frame0_2.shape[1], 3))
    # H2 = H2.repeat(4, axis=0).repeat(4, axis=1)
    # Y2 = Y2.repeat(4, axis=0).repeat(4, axis=1)
    # H2 = H2.reshape((-1, 3, 2))
    # Y2 = Y2.reshape((-1, 3))
    # H = np.append(H0, H2, axis=1)
    # Y = np.append(Y0, Y2, axis=1)

    # scale = 0.5
    # frame0_2 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # frame1_2 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # H2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D, Global.D))
    # Y2 = np.zeros((frame0_2.shape[0] * frame0_2.shape[1], Global.C_D))
    #
    # [Ix, Iy, It] = Compute_Gradient(frame0_2, frame1_2)
    #
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H2[:, :, 0] = Ix
    # H2[:, :, 1] = Iy
    # Y2[:, :] = -It * 0.5
    #
    # H = np.append(H, H2, axis=0)
    # Y = np.append(Y, Y2, axis=0)
    # scale = 0.25
    # frame0_3 = cv2.resize(Global.frame0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # frame1_3 = cv2.resize(frame1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # H3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D, Global.D))
    # Y3 = np.zeros((frame0_3.shape[0] * frame0_3.shape[1], Global.C_D))
    #
    # [Ix, Iy, It] = Compute_Gradient(frame0_3, frame1_3)
    #
    # Ix = np.reshape(Ix, (-1, Global.C_D))
    # Iy = np.reshape(Iy, (-1, Global.C_D))
    # It = np.reshape(It, (-1, Global.C_D))
    # H3[:, :, 0] = Ix
    # H3[:, :, 1] = Iy
    # Y3[:, :] = -It * 0.25
    #
    # H = np.append(H, H3, axis=0)
    # Y = np.append(Y, Y3, axis=0)

    # plt.subplot(326)
    mu_max, clusters, Cmax = KmeansTF(H, Y)
    print(Back.RED)
    print("Number of clusters: ", Global.K_C)
    for i in range(0, Global.K_C + 1):
        print("Estimate Mu[", i, "] :", mu_max[i], "C: ", Cmax[i])
    print(Style.RESET_ALL)
    #
    # plt.subplot(325)
    # for k in range(0, Global.K_C + 1):
    #     xk = np.asarray([X[j, :] for j in range(Global.N) if clusters[j] == k])
    #     if (xk.size):
    #         color = Global.colors[k]
    #         my_plot.PlotPoints(xk, color, 'X' + str(k) + '- Data')


def ArttificalFlowK():
    """Creates articifal flow and clusters it using soft K-means.

    **Parameters**:
     -

    **Returns**:
     - Figure with the clusterd points
    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.
    .. note:: Can use real flow by adding the comment line in the function
    """
    Y = np.zeros((Global.N, Global.C_D))
    H = np.zeros((Global.N, Global.C_D, Global.D))
    Xk, X, U, V, MU, SIGMA, C, framePoints = my_sample.SampleXK()
    print(Back.BLUE)
    print("Number of clusters: ", Global.K)
    for i in range(0, Global.K):
        print("Real Mu[", i, "] :", MU[i], "Real C[", i, "]: ", C[i])
    print(Style.RESET_ALL)

    maxXY = np.amax([np.amax(X[0, :]), np.amax(X[1, :])])
    minXY = np.amin([np.amin(X[0, :]), np.amin(X[1, :])])

    # Set Axes
    fig, ax = plt.subplots(3, 2)
    for (m, n), subplot in np.ndenumerate(ax):
        subplot.set_xlim(minXY, maxXY)
        subplot.set_ylim(minXY, maxXY)
        plt.axis("equal")

    yy, xx = np.mgrid[:Global.HEIGHT, :Global.WIDTH]
    xx = xx - U
    yy = yy - V
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    plt.clf()
    ax = plt.subplot(321)
    ax.set_title("Frame 0")
    plt.imshow(cv2.cvtColor(Global.frame0, cv2.COLOR_BGR2RGB))
    frame1 = cv2.remap(Global.frame0, xx, yy, cv2.INTER_CUBIC)

    # frame1 = cv2.imread(Global.IMAGE2) #Add this line in order to use 2 frames
    # frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    ax = plt.subplot(322)
    ax.set_title("Frame 1")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

    ax = plt.subplot(323)
    ax.set_title("Real Points and contour")

    for i in range(0, Global.K):
        color = Global.colors[i + 1]
        my_plot.PlotContour(MU[i], SIGMA[i], Xk[i], color, 'Original-' + str(i))
        my_plot.PlotPoints(Xk[i], color, 'X' + str(i) + '- Data')

    ax = plt.subplot(324)
    framePoints = framePoints.astype('uint8')
    plt.imshow(framePoints)

    [Ix, Iy, It] = Compute_Gradient(Global.frame0, frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))

    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It
    plt.subplot(326)
    mu_max, clusters, Cmax = Kmeans(H, Y)
    print(Back.RED)
    print("Number of clusters: ", Global.K_C)
    for i in range(00, Global.K_C + 1):
        print("Estimate Mu[", i, "] :", mu_max[i], "C: ", Cmax[i])
    print(Style.RESET_ALL)

    plt.subplot(325)
    for k in range(0, Global.K_C + 1):
        xk = np.asarray([X[j, :] for j in range(Global.N) if clusters[j] == k])
        if (xk.size):
            color = Global.colors[k]
            my_plot.PlotPoints(xk, color, 'X' + str(k) + '- Data')


def ArttificalFlowK2():
    """Creates articifal flow and clusters it using soft K-means.

    **Parameters**:
     -

    **Returns**:
     - Figure with the clusterd points
    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.
    .. note:: Can use real flow by adding the comment line in the function
    """
    Y = np.zeros((Global.N, Global.C_D))
    H = np.zeros((Global.N, Global.C_D, Global.D))

    Xk, X, U, V, MU, SIGMA, C, framePoints = my_sample.SampleXK()
    print(Back.BLUE)
    print("Number of clusters: ", Global.K)
    for i in range(0, Global.K):
        print("Real Mu[", i, "] :", MU[i], "Real C[", i, "]: ", C[i])
    print(Style.RESET_ALL)

    maxXY = np.amax([np.amax(X[0, :]), np.amax(X[1, :])])
    minXY = np.amin([np.amin(X[0, :]), np.amin(X[1, :])])

    # Set Axes
    fig, ax = plt.subplots(3, 2)
    for (m, n), subplot in np.ndenumerate(ax):
        subplot.set_xlim(minXY, maxXY)
        subplot.set_ylim(minXY, maxXY)
        plt.axis("equal")

    yy, xx = np.mgrid[:Global.HEIGHT, :Global.WIDTH]
    xx = xx - U
    yy = yy - V
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    plt.clf()
    """ For subplots
    ax = plt.subplot(321)
    ax.set_title("Frame 0")
    plt.imshow(cv2.cvtColor(Global.frame0, cv2.COLOR_BGR2RGB))
    For subplots"""
    frame1 = cv2.remap(Global.frame0, xx, yy, cv2.INTER_CUBIC)

    frame1 = cv2.imread(Global.IMAGE2)  # Add this line in order to use 2 frames
    # frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    """ For subplots
    ax = plt.subplot(322)
    ax.set_title("Frame 1")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    For subplots """

    """ For subplots

    ax = plt.subplot(323)
    ax.set_title("Real Points and contour")

    # for i in range(0, Global.K):
    #     color = Global.colors[i + 1]
    #     my_plot.PlotContour(MU[i], SIGMA[i], Xk[i], color, 'Original-' + str(i))
    #     my_plot.PlotPoints(Xk[i], color, 'X' + str(i) + '- Data')

    ax = plt.subplot(324)
    framePoints = framePoints.astype('uint8')
    plt.imshow(framePoints)
    for subplots"""

    [Ix, Iy, It] = Compute_Gradient(Global.frame0, frame1)
    Ix = np.reshape(Ix, (-1, Global.C_D))
    Iy = np.reshape(Iy, (-1, Global.C_D))
    It = np.reshape(It, (-1, Global.C_D))

    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It

    # plt.subplot(326)
    # mu_max, clusters, Cmax = Kmeans(H, Y)
    mu_max, clusters, Cmax = KmeansTF(H, Y)
    print(Back.RED)
    print("Number of clusters: ", Global.K_C)
    for i in range(00, Global.K_C + 1):
        print("Estimate Mu[", i, "] :", mu_max[i], "C: ", Cmax[i])
    print(Style.RESET_ALL)

    # plt.subplot(325)
    # for k in range(0, Global.K_C + 1):
    #     xk = np.asarray([X[j, :] for j in range(Global.N) if clusters[j] == k])
    #     if (xk.size):
    #         color = Global.colors[k]
    #         my_plot.PlotPoints(xk, color, 'X' + str(k) + '- Data')


def OpticalFlowK():
    """Creates articifal flow and clusters it using soft K-means.

    **Parameters**:
     -

    **Returns**:
     - Figure with the clusterd points

    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.
    .. note:: Can use real flow by adding the comment line in the function"""
    global MU
    global SIGMA
    global C

    Y = np.zeros((N, C_D))
    H = np.zeros((N, C_D, D))

    # Xk,X,U,V,MU,SIGMA,C,framePoints=my_sample.SampleXK()

    """START
#    
#    print(Back.BLUE)
#    print("Number of clusters: " , K)
#    for i in range (0,K):
#        print("Real Mu[",i,"] :" , MU[i] , "Real C[",i,"]: " ,C[i]) 
#    print(Style.RESET_ALL)
#    
#    maxXY=np.amax([np.amax(X[0,:]),np.amax(X[1,:])])
#    minXY=np.amin([np.amin(X[0,:]),np.amin(X[1,:])])
#        
#
#    # Set Axes
#    fig, ax = plt.subplots(3,2)
#    for (m,n), subplot in np.ndenumerate(ax):
#        subplot.set_xlim(minXY,maxXY)
#        subplot.set_ylim(minXY,maxXY)
#        plt.axis("equal")
#    
#   
#    yy,xx=np.mgrid[:HEIGHT,:WIDTH]
#    xx=xx-U
#    yy=yy-V 
#    xx=xx.astype(np.float32)
#    yy=yy.astype(np.float32)
#    
#    plt.clf()
#    ax=plt.subplot(321)
#    ax.set_title("Frame 0")
#    plt.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
#    frame1=cv2.remap(frame0,xx,yy,cv2.INTER_CUBIC)
#    ax=plt.subplot(322)
#    ax.set_title("Frame 1")
#    plt.imshow(frame1)
    END"""
    [Ix, Iy, It] = Compute_Gradient(frame0, frame1)
    Ix = np.reshape(Ix, (-1, C_D))
    Iy = np.reshape(Iy, (-1, C_D))
    It = np.reshape(It, (-1, C_D))

    """ START
    ax=plt.subplot(323)
    ax.set_title("Real Points and contour")

    for i in range(0,K):
        color=colors[i+1]
        my_plot.PlotContour(MU[i],SIGMA[i],Xk[i],color, 'Original-'+str(i))
        my_plot.PlotPoints(Xk[i],color,'X'+str(i)+'- Data')

    ax=plt.subplot(324)
    framePoints=framePoints.astype('uint8')
    plt.imshow(framePoints)
    END"""

    H[:, :, 0] = Ix
    H[:, :, 1] = Iy
    Y[:, :] = -It

    #    plt.subplot(326)
    mu_max, clusters = Kmeans(H, Y)


#    print(Back.RED)
#    print("Number of clusters: " , K_C)
#    for i in range (0,K_C):
#        print("Estimate Mu[",i,"] :" , mu_max[i] )
#    print(Style.RESET_ALL)
#
#    plt.subplot(325)
#    for k in range (0,K_C+1):
#        xk=np.asarray([X[j,:] for j in range(N) if clusters[j] == k])
#        if(xk.size):
#            color=colors[k]
#            my_plot.PlotPoints(xk,color,'X'+str(k)+'- Data')
#


"""
************************************************************************************************************************************************************

                                                                        K-Means Functions

************************************************************************************************************************************************************
"""


def KmeansSPCuda(X, H, Y):
    global SIGMA
    global SIGMA1
    global SIGMA2
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    SIGMA2 = np.array([[Global.loc_scale, 0., 0., 0., 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0., 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0., 0., 0., 0.],
                       [0., 0., 0., Global.int_scale, 0., 0., 0., 0.],
                       [0., 0., 0., 0., Global.int_scale, 0., 0., 0.],
                       [0., 0., 0., 0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., 0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., 0., 0., 0., Global.int_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 600
    it = 0
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = Z / (Zmax)

    threshold = np.percentile(Z,
                              5)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0

    arr = np.array([1, 2, 2, 2], dtype=np.int32)
    adder = cuda_sp.cuda_SP(arr)
    adder.increment()
    X = X.astype(np.float32)
    H = H.astype(np.float32)
    Y = Y.astype(np.float32)
    r_ik = r_ik.astype(np.float32)
    pi = pi.astype(np.float32)
    SIGMA = SIGMA.astype(np.float32)
    SIGMA1 = SIGMA1.astype(np.float32)
    SIGMA2 = SIGMA2.astype(np.float32)
    X = X.copy(order='C')
    H = H.copy(order='C')
    Y = Y.copy(order='C')
    r_ik = r_ik.copy(order='C')
    pi = pi.copy(order='C')
    SIGMA1 = np.array([Global.loc_scale, 0.0000001, 0.0000001, Global.loc_scale, Global.int_scale, Global.int_scale,
                       Global.int_scale], dtype=np.float32)
    SIGMA1 = 1.0 / SIGMA1
    SIGMA2 = np.array([Global.opt_scale, Global.opt_scale, Global.opt_scale], dtype=np.float32)
    SIGMA2 = 1.0 / SIGMA2;
    SIGMA1[1] = 0
    SIGMA1[2] = 0
    SIGMA1 = SIGMA1.copy(order='C')
    SIGMA2 = SIGMA2.copy(order='C')
    LOOKUP_TABLE_CUDA = Global.LOOKUP_TABLE_CUDA.astype(np.int32)
    LOOKUP_TABLE_CUDA = LOOKUP_TABLE_CUDA.copy(order='C')
    idx_pixels = Global.idx_pixels_cuda  # .cpu().numpy()
    idx_pixels = idx_pixels.astype(np.int32)
    idx_pixels = idx_pixels.copy(order='C')

    print("Before adder")
    XData2 = adder.KmeansC2(X.reshape(-1), H.reshape(-1), Y.reshape(-1), r_ik.reshape(-1), pi.reshape(-1),
                            SIGMA1.reshape(-1), SIGMA2.reshape(-1), idx_pixels.reshape(-1),
                            LOOKUP_TABLE_CUDA.reshape(-1))
    print("After adder")
    XData2 = np.asarray(XData2)
    r_ik = XData2.reshape(-1, 17)
    # r_ik_t = torch.from_numpy(XData2).contiguous().cuda().float()
    # r_ik = r_ik_t.cpu().numpy()
    r_ik = np.argmax(r_ik, axis=1)
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]
    fig = plt.figure()
    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    fig.savefig('plot.svg', format='svg', dpi=1200)


def KmeansSP2(X, H, Y):
    """Soft K-means algorithem over the location,intesity and the optical flow

    **Parameters**:
     - :math:`X[N,5]` - Data matrix  [Point number,(X,Y,L,A,B].
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].
     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    global SIGMA
    global SIGMA1
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 50
    it = 0
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = Z / (Zmax)
    # Z[Z==0]=np.nan
    #    Z=np.prod(Z,axis=1)
    threshold = np.percentile(Z,
                              5)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0
    while (it < maxIt):
        it += 1
        C1 = EstimateSP(X, r_ik, pi)  # M-Step
        C2, _, _ = EstimateParams(H, Y, r_ik, norm_Weight)
        C = np.append(C1, C2, axis=1)
        Nk = r_ik.sum(0)
        N_ = X.shape[1]
        pi = Nk / N_

        _, r_ik = FindClosestClusterSP2(X, C1, C2, pi, r_ik, H, Y, Global.opt_scale, norm_Weight)  # E-Step

    #    r_ik=np.argmax(r_ik,axis=1)
    #    r_ik2=np.reshape(r_ik,(HEIGHT,WIDTH))
    #    framePointsNew=np.zeros((HEIGHT,WIDTH,3))
    #    for i in range(0,HEIGHT):
    #        for j in range(0,WIDTH):
    #            if(r_ik2[i,j]==-1):
    #                framePointsNew[i,j,:]=0,0,0
    #            if(r_ik2[i,j]==-2):
    #                framePointsNew[i,j,:]=255,255,255
    #            if(r_ik2[i,j]>=0):
    #                ind=int(r_ik2[i,j])
    #                framePointsNew[i,j,:]=my_help.hex2rgb(colors[ind])
    #
    #    framePointsNew=framePointsNew.astype('uint8')
    #    plt.imshow(framePointsNew)
    r_ik = np.argmax(r_ik, axis=1)
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):

            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]

    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    return C, r_ik


def KmeansSP2TF(X, H, Y):
    """Soft K-means algorithem over the location,intesity and the optical flow

    **Parameters**:
     - :math:`X[N,5]` - Data matrix  [Point number,(X,Y,L,A,B].
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].
     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    global SIGMA
    global SIGMA1
    global SIGMA2
    global SIGMAxylab
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    SIGMA2 = np.array([[Global.loc_scale, 0., 0., 0., 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0., 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0., 0., 0., 0.],
                       [0., 0., 0., Global.int_scale, 0., 0., 0., 0.],
                       [0., 0., 0., 0., Global.int_scale, 0., 0., 0.],
                       [0., 0., 0., 0., 0., Global.opt_scale, 0., 0.],
                       [0., 0., 0., 0., 0., 0., Global.opt_scale, 0.],
                       [0., 0., 0., 0., 0., 0., 0., Global.opt_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 500
    it = 0
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)

    threshold = np.percentile(Z,0.00001)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    maskall=np.all(np.isnan(Z),axis=1)
    mask0=np.isnan(Z[:,0])
    mask1=np.isnan(Z[:,1])
    mask2=np.isnan(Z[:,2])

    H[~maskall & mask0 & mask1, 0] = H[~maskall & mask0 & mask1, 2]
    H[~maskall & mask0 & mask2, 0] = H[~maskall & mask0 & mask2, 1]
    H[~maskall & mask1 & mask0, 1] = H[~maskall & mask1 & mask0, 2]
    H[~maskall & mask1 & mask2, 1] = H[~maskall & mask1 & mask2, 0]
    H[~maskall & mask2 & mask0, 2] = H[~maskall & mask2 & mask0, 1]
    H[~maskall & mask2 & mask1, 2] = H[~maskall & mask2 & mask1, 0]

    Y[~maskall & mask0 & mask1, 0] = Y[~maskall & mask0 & mask1, 2]
    Y[~maskall & mask0 & mask2, 0] = Y[~maskall & mask0 & mask2, 1]
    Y[~maskall & mask1 & mask0, 1] = Y[~maskall & mask1 & mask0, 2]
    Y[~maskall & mask1 & mask2, 1] = Y[~maskall & mask1 & mask2, 0]
    Y[~maskall & mask2 & mask0, 2] = Y[~maskall & mask2 & mask0 , 1]
    Y[~maskall & mask2 & mask1, 2] = Y[~maskall & mask2 & mask1, 0]

    maskall = torch.from_numpy((np.argwhere(np.all(np.isnan(Z), axis=1))[:,0]+0)).cuda()
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z


    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0



    print("CPU->GPU")

    argmax=torch.from_numpy(np.argmax(r_ik,axis=1)).cuda()
    H_t = torch.from_numpy(H).cuda().float()
    H_t5=H_t.reshape(-1, 1, 2).repeat(1, 5, 1)
    Y_t = torch.from_numpy(Y).cuda().unsqueeze(2).float()
    X_t = torch.from_numpy(X).cuda().float()
    SIGMA = torch.from_numpy(SIGMA).cuda().float()
    SIGMA1 = torch.from_numpy(SIGMA1).cuda().float()
    SIGMA2 = torch.from_numpy(SIGMA2).cuda().float()
    T1_temp = torch.bmm(H_t.transpose(1, 2), Y_t)[:,:,0]
    HH = torch.bmm(H_t.transpose(1, 2), H_t).reshape(-1,4)
    YY = torch.bmm(Y_t.transpose(1, 2), Y_t)[:,:,0]
    index2_buffer = torch.zeros(Global.N).cuda()
    alpha_prime = Global.ALPHA_T + (Global.D_T * Global.N) / (2.0)
    eta_prime = torch.zeros(Global.K_C + 1, 2).cuda()
    H_t = H_t.reshape(-1, 2)
    Y_t = Y_t.reshape(1, -1)
    r_ikNew_buffer = torch.zeros((Global.N, 5)).cuda().reshape(-1)
    range1=torch.arange(0, Global.N * 25).cuda()
    range2=torch.arange(0, Global.N * 5*2).cuda()
    range3=torch.arange(0, Global.N*5).cuda()
    range4=torch.arange(0, Global.N*5*4).cuda()

    range_conn=(torch.arange(Global.N)*5).cuda()
    c1_temp = torch.zeros((Global.N * 5 * 5)).cuda()
    c2_temp = torch.zeros((Global.N * 5 * 2)).cuda()
    pi_temp = torch.zeros((Global.N*5)).cuda()
    SigmaXY_temp=torch.zeros((Global.N*5*4)).cuda().float()
    logdet_temp=torch.zeros((Global.N*5)).cuda()
    SigmaXY=torch.zeros((Global.K_C+1,4)).cuda().float()
    SigmaXY_i=torch.zeros((Global.K_C+1,4)).cuda().float()
    SIGMAxylab = torch.zeros((Global.K_C + 1, 6,6)).cuda().float()
    SIGMAxylab[:,2,2]=Global.int_scale
    SIGMAxylab[:,3,3]=Global.int_scale
    SIGMAxylab[:,4,4]=Global.int_scale
    SIGMAxylab[:,5,5]=Global.opt_scale
    opt_scale=torch.from_numpy(np.array([Global.opt_scale],dtype=np.float32)).cuda()
    int_scale=torch.from_numpy(np.array([Global.int_scale],dtype=np.float32)).cuda()


    Nk = torch.zeros(Global.K_C + 1).float().cuda()
    X1 = torch.zeros(Global.K_C + 1, 5).float().cuda()
    X2_00 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_01 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_11 = torch.zeros(Global.K_C + 1).float().cuda()


    T1 = torch.zeros(Global.K_C+1,2).float().cuda()
    T2 = torch.zeros(Global.K_C + 1, 4).float().cuda()
    T3 = torch.zeros(Global.K_C + 1).float().cuda()

    X_C_SIGMA=torch.zeros(Global.N,5,5).float().cuda()
    distances1=torch.zeros(Global.N,5).float().cuda()
    r_ik_5=torch.zeros(Global.N,5).float().cuda()
    sum_buffer=torch.zeros(Global.N).float().cuda()


    while (it < maxIt):
        it += 1
        if (it % 20 == 0):
            print("It :",it)
        C1,SigmaXY,SigmaXY_i,Nk= EstimateSPTF(X_t, argmax, SigmaXY,SigmaXY_i,Nk,X1,X2_00,X2_01,X2_11)  # M-Step
        C2, _, _ = TFEstimateParams(H_t, Y_t, argmax, T1_temp, HH, YY, alpha_prime, eta_prime,T1,T2,T3)
        C2[0, 0] = 100
        C2[0, 1] = 100
        #C = torch.cat([C1, C2], dim=1)
        N_ = X.shape[0]
        pi_t = torch.div(torch.mul(Nk, (1 - Global.PI_0)) , (N_ - Nk[0]))
        pi_t[0] = Global.PI_0_T

        prev_r_ik_max=argmax.clone()
        c_idx=prev_r_ik_max.view(-1).index_select(0,Global.c_idx)

        prev_r_ik_max = prev_r_ik_max.view((Global.HEIGHT, Global.WIDTH))

        c1_vals=C1.index_select(0,c_idx).view(-1)
        c2_vals=C2.index_select(0,c_idx).view(-1)
        pi_vals=pi_t.index_select(0,c_idx).view(-1)
        SigmaXY_vals=SigmaXY_i.index_select(0,c_idx).view(-1)
        _, logdet = np.linalg.slogdet(SIGMAxylab)
        logdet = torch.from_numpy(logdet).cuda()
        logdet_vals=logdet.index_select(0,c_idx).view(-1)
        c1_temp.scatter_(0, range1, c1_vals)
        c2_temp.scatter_(0, range2, c2_vals)
        pi_temp.scatter_(0,range3,pi_vals)
        logdet_temp.scatter_(0,range3,logdet_vals)

        SigmaXY_temp.scatter_(0,range4,SigmaXY_vals)

        FindClosestClusterSP2TF(X_t, Y_t, opt_scale,int_scale ,logdet_temp, H_t5,c1_temp,c2_temp,pi_temp,SigmaXY_temp.reshape(-1,5,4),X_C_SIGMA,distances1,r_ik_5,sum_buffer,maskall)  # E-Step


        argmax= my_connectivity.Change_pixel(prev_r_ik_max, r_ik_5, index2_buffer, r_ikNew_buffer, it % 4,c_idx,_,range_conn)

    print("GPU->CPU")

    r_ik=argmax.cpu().numpy()
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))

    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    framePointsNew2= np.zeros((Global.HEIGHT, Global.WIDTH, 3))


    mean_value = np.zeros((Global.K_C + 1, 3))
    mean_value2 = np.zeros((Global.K_C + 1, 3))

    C2=C2.cpu().numpy()
    C2[0,0]=0
    C2[np.argwhere(np.isnan(C2))]=0
    C2=np.expand_dims(C2,1)
    C2=C2.reshape(-1,1,2)
    flow=my_help.computeImg(C2)
    #
    # hsv = np.zeros((Global.K_C+1,1,3), dtype=np.uint8)
    # hsv[..., 1] = 255
    #
    # mag, ang = cv2.cartToPolar(C2[..., 0], C2[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    #bgr = cv2.c0vtColor(hsv, cv2.COLOR_HSV2BGR)
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)
        mean_value2[i,0] =flow[i,:,0]
        mean_value2[i,1] =flow[i,:,1]
        mean_value2[i,2] =flow[i,:,2]

    framePointsNew3 = Global.frame0

    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if(i==0):
                if(j==0):
                    if((r_ik2[i,j]==r_ik2[i+1,j])and(r_ik2[i,j]==r_ik2[i,j+1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]
                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0

                elif(j==Global.WIDTH-1):
                    if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j -1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]
                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0

                else:
                    if((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])and(r_ik2[i,j]==r_ik2[i,j+1]) ):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0



            elif(i==Global.HEIGHT-1):
                if (j == 0):
                    if ((r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0

                elif (j == Global.WIDTH-1):
                    if ((r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                else:
                    if ((r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0



            else:
                if (j == 0):
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                elif (j == Global.WIDTH-1):
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                else:
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


    fig = plt.figure()
    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_Avg.png', format='png', dpi=1200)
    fig = plt.figure()
    framePointsNew2 = framePointsNew2.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_White.png', format='png', dpi=1200)
    fig = plt.figure()
    framePointsNew3 = framePointsNew3.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_Original.png', format='png', dpi=1200)

    return [C1,C2], r_ik

def Kmeans2Frames(X,loc):
    global SIGMA
    global SIGMA1
    global SIGMA2
    global SIGMAxylab
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    SIGMA2 = np.array([[Global.loc_scale, 0., 0., 0., 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0., 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0., 0., 0., 0.],
                       [0., 0., 0., Global.int_scale, 0., 0., 0., 0.],
                       [0., 0., 0., 0., Global.int_scale, 0., 0., 0.],
                       [0., 0., 0., 0., 0., Global.opt_scale, 0., 0.],
                       [0., 0., 0., 0., 0., 0., Global.opt_scale, 0.],
                       [0., 0., 0., 0., 0., 0., 0., Global.opt_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 450
    it = 0
    print("CPU->GPU")

    argmax = torch.from_numpy(np.argmax(r_ik, axis=1)).cuda()
    X = torch.from_numpy(X).cuda().float()
    SIGMA = torch.from_numpy(SIGMA).cuda().float()
    SIGMA1 = torch.from_numpy(SIGMA1).cuda().float()
    SIGMA2 = torch.from_numpy(SIGMA2).cuda().float()
    loc = torch.from_numpy(loc).cuda().float()


    index2_buffer = torch.zeros(Global.N).cuda()

    r_ikNew_buffer = torch.zeros((Global.N, 5)).cuda().reshape(-1)

    range1 = torch.arange(0, Global.N * 12*5).cuda()
    range3 = torch.arange(0, Global.N * 5).cuda()
    range4 = torch.arange(0, Global.N * 5 * 8).cuda()

    range_conn = (torch.arange(Global.N) * 5).cuda()
    c1_temp = torch.zeros((Global.N * 12 * 5)).cuda()
    pi_temp = torch.zeros((Global.N * 5)).cuda()

    SigmaXY_temp = torch.zeros((Global.N * 5 * 8)).cuda().float()
    logdet_temp = torch.zeros((Global.N * 5)).cuda()
    SigmaXY = torch.zeros((Global.K_C + 1, 8)).cuda().float()
    SigmaXY_i = torch.zeros((Global.K_C + 1, 8)).cuda().float()

    SIGMAxylab = torch.zeros((Global.K_C + 1, 12,12)).cuda().float()
    SIGMAxylab[:, 2, 2] = Global.int_scale
    SIGMAxylab[:, 3, 3] = Global.int_scale
    SIGMAxylab[:, 4, 4] = Global.int_scale
    SIGMAxylab[:, 7, 7] = Global.int_scale
    SIGMAxylab[:, 8, 8] = Global.int_scale
    SIGMAxylab[:, 9, 9] = Global.int_scale
    SIGMAxylab[:, 10, 10] = Global.opt_scale
    SIGMAxylab[:, 11, 11] = Global.opt_scale


    opt_scale = torch.from_numpy(np.array([Global.opt_scale], dtype=np.float32)).cuda()
    int_scale = torch.from_numpy(np.array([Global.int_scale], dtype=np.float32)).cuda()

    Nk = torch.zeros(Global.K_C + 1).float().cuda()
    X1 = torch.zeros(Global.K_C + 1, 12).float().cuda()

    X2_00 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_01 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_11 = torch.zeros(Global.K_C + 1).float().cuda()
    X_C_SIGMA = torch.zeros(Global.N, 5, 12).float().cuda()
    distances1 = torch.zeros(Global.N, 5).float().cuda()
    r_ik_5 = torch.zeros(Global.N, 5).float().cuda()
    sum_buffer = torch.zeros(Global.N).float().cuda()
    c_idx=0
    init=True

    while (it < maxIt):
        it += 1
        if (it % 20 == 0):
            print("It :", it)
        C1, SigmaXY, SigmaXY_i, Nk = EstimateSP_2Frames(X,loc, argmax, SigmaXY, SigmaXY_i, Nk, X1, X2_00, X2_01,X2_11,r_ik_5,c_idx,init)  # M-Step
        N_ = X.shape[0]
        pi_t = torch.div(torch.mul(Nk, (1 - Global.PI_0)), (N_ - Nk[0]))
        pi_t[0] = Global.PI_0_T

        if (Global.HARD_EM == True or init==True):
            prev_r_ik_max = argmax.clone()
            init = False


        c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)

        prev_r_ik_max = prev_r_ik_max.view((Global.HEIGHT, Global.WIDTH))

        c1_vals = C1.index_select(0, c_idx).view(-1)
        pi_vals = pi_t.index_select(0, c_idx).view(-1)
        SigmaXY_vals = SigmaXY_i.index_select(0, c_idx).view(-1)
        _, logdet = np.linalg.slogdet(SIGMAxylab)
        logdet = torch.from_numpy(logdet).cuda()
        logdet_vals = logdet.index_select(0, c_idx).view(-1)
        c1_temp.scatter_(0, range1, c1_vals)
        pi_temp.scatter_(0, range3, pi_vals)
        logdet_temp.scatter_(0, range3, logdet_vals)
        SigmaXY_temp.scatter_(0, range4, SigmaXY_vals)

        r_ik_5=FindClosestClusterSP_2Frames(X, int_scale, opt_scale, logdet_temp, c1_temp, pi_temp,  SigmaXY_temp.reshape(-1, 5, 8), X_C_SIGMA,distances1,r_ik_5, sum_buffer) #TODO: Check  r_ik_5 pointer

        if(Global.HARD_EM==True):
            argmax = my_connectivity.Change_pixel(prev_r_ik_max, r_ik_5, index2_buffer, r_ikNew_buffer, it % 4, c_idx, _,range_conn)
        else:
            prev_r_ik_max = r_ik_5.argmax(1)
            prev_r_ik_max=torch.take(c_idx,torch.add(prev_r_ik_max,range_conn))
    pr.disable()
    print("GPU->CPU")


    r_ik=argmax.cpu().numpy()
    if (Global.HARD_EM==False):
        prev_r_ik_max = r_ik_5.argmax(1)
        prev_r_ik_max = torch.take(c_idx, torch.add(prev_r_ik_max, range_conn))
        r_ik=prev_r_ik_max.cpu().numpy()

    SigmaXY_cpu=SigmaXY.cpu().numpy()[:,0:4]
    r_ik2= np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    C1_cpu=C1.cpu().numpy()[:,0:2]

    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    framePointsNew2= np.zeros((Global.HEIGHT, Global.WIDTH, 3))


    mean_value = np.zeros((Global.K_C + 1, 3))
    mean_value2 = np.zeros((Global.K_C + 1, 3))

    Trueflow=np.expand_dims(C1[:,10:12],1)
    Trueflow=Trueflow.reshape(-1,1,2)
    flow=my_help.computeImg(Trueflow)
    #
    # hsv = np.zeros((Global.K_C+1,1,3), dtype=np.uint8)
    # hsv[..., 1] = 255
    #
    # mag, ang = cv2.cartToPolar(C2[..., 0], C2[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)
        mean_value2[i,0] =flow[i,:,0]
        mean_value2[i,1] =flow[i,:,1]
        mean_value2[i,2] =flow[i,:,2]

    np.save(str(Global.IMAGE1)+'_mean_RGB',mean_value)
    np.save(str(Global.IMAGE1)+'_mean_XY',C1_cpu)
    np.save(str(Global.IMAGE1)+'_SigmaXY',SigmaXY_cpu)
    np.save(str(Global.IMAGE1)+'_Clusters',r_ik2)

    framePointsNew3 = Global.frame0

    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if(i==0):
                if(j==0):
                    if((r_ik2[i,j]==r_ik2[i+1,j])and(r_ik2[i,j]==r_ik2[i,j+1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]
                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0

                elif(j==Global.WIDTH-1):
                    if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j -1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]
                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0

                else:
                    if((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])and(r_ik2[i,j]==r_ik2[i,j+1]) ):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0



            elif(i==Global.HEIGHT-1):
                if (j == 0):
                    if ((r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0

                elif (j == Global.WIDTH-1):
                    if ((r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                else:
                    if ((r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0



            else:
                if (j == 0):
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                elif (j == Global.WIDTH-1):
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


                else:
                    if ((r_ik2[i,j]==r_ik2[i+1,j])and (r_ik2[i, j] == r_ik2[i -1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
                        ind = int(r_ik2[i, j])
                        framePointsNew[i, j, :] = mean_value[ind]
                        framePointsNew2[i, j, :] = mean_value2[ind]

                    else:
                        framePointsNew[i, j, :] = 0
                        framePointsNew2[i, j, :] = 0
                        framePointsNew3[i, j, :] = 0


    fig = plt.figure()
    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_Avg.png', format='png', dpi=1200)
    fig = plt.figure()
    framePointsNew2 = framePointsNew2.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_White.png', format='png', dpi=1200)
    fig = plt.figure()
    framePointsNew3 = framePointsNew3.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
    fig.savefig('frame1_'+str(Global.K_C)+'SP_Original.png', format='png', dpi=1200)



def KmeansSplitMerge(X,loc):
    global SIGMA
    global SIGMA1
    global SIGMA2
    global SIGMAxylab
    global SIGMAxylab_s
    _=1
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    SIGMA2 = np.array([[Global.loc_scale, 0., 0., 0., 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0., 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0., 0., 0., 0.],
                       [0., 0., 0., Global.int_scale, 0., 0., 0., 0.],
                       [0., 0., 0., 0., Global.int_scale, 0., 0., 0.],
                       [0., 0., 0., 0., 0., Global.opt_scale, 0., 0.],
                       [0., 0., 0., 0., 0., 0., Global.opt_scale, 0.],
                       [0., 0., 0., 0., 0., 0., 0., Global.opt_scale]])


    X = torch.from_numpy(X).cuda().float()
    SIGMA = torch.from_numpy(SIGMA).cuda().float()
    SIGMA1 = torch.from_numpy(SIGMA1).cuda().float()
    SIGMA2 = torch.from_numpy(SIGMA2).cuda().float()
    loc = torch.from_numpy(loc).cuda().float()


    index2_buffer = torch.zeros(Global.N).cuda()

    r_ikNew_buffer = torch.zeros((Global.N, 5)).cuda().reshape(-1)

    range1 = torch.arange(0, Global.N * Global.D_*Global.neig_num).cuda()
    range3 = torch.arange(0, Global.N * Global.neig_num).cuda()
    range4 = torch.arange(0, Global.N * Global.neig_num * Global.D_Inv).cuda()

    range_conn = (torch.arange(Global.N) * Global.neig_num).cuda()

    c1_temp = torch.zeros((Global.N * Global.D_* Global.neig_num)).cuda()

    pi_temp = torch.zeros((Global.N * Global.neig_num)).cuda()

    SigmaXY_temp = torch.zeros((Global.N * Global.neig_num * Global.D_Inv)).cuda().float()
    logdet_temp = torch.zeros((Global.N * Global.neig_num)).cuda()
    SigmaXY = torch.zeros((Global.K_C + 1, Global.D_Inv)).cuda().float()
    SigmaXY_i = torch.zeros((Global.K_C + 1, Global.D_Inv)).cuda().float()

    SigmaXY_s = torch.zeros(((Global.K_C + 1)*2, Global.D_Inv)).cuda().float()
    SigmaXY_i_s = torch.zeros((Global.K_C + 1)*2, Global.D_Inv).cuda().float()


    SIGMAxylab = torch.zeros((Global.K_C + 1, Global.D_,Global.D_)).cuda().float()
    SIGMAxylab[:, 2, 2] = Global.int_scale
    SIGMAxylab[:, 3, 3] = Global.int_scale
    SIGMAxylab[:, 4, 4] = Global.int_scale
    if(Global.D_12):
        SIGMAxylab[:, 7, 7] = Global.int_scale
        SIGMAxylab[:, 8, 8] = Global.int_scale
        SIGMAxylab[:, 9, 9] = Global.int_scale
        SIGMAxylab[:, 10, 10] = Global.opt_scale
        SIGMAxylab[:, 11, 11] = Global.opt_scale

    SIGMAxylab_s = torch.zeros(((Global.K_C + 1)*2, Global.D_,Global.D_)).cuda().float()
    SIGMAxylab_s[:, 2, 2] = Global.int_scale
    SIGMAxylab_s[:, 3, 3] = Global.int_scale
    SIGMAxylab_s[:, 4, 4] = Global.int_scale
    if(Global.D_12):
        SIGMAxylab_s[:, 7, 7] = Global.int_scale
        SIGMAxylab_s[:, 8, 8] = Global.int_scale
        SIGMAxylab_s[:, 9, 9] = Global.int_scale
        SIGMAxylab_s[:, 10, 10] = Global.opt_scale
        SIGMAxylab_s[:, 11, 11] = Global.opt_scale

    Nk = torch.zeros(Global.K_C + 1).float().cuda()
    Nk_s= torch.zeros((Global.K_C + 1)*2).float().cuda()
    X1 = torch.zeros(Global.K_C + 1, Global.D_).float().cuda()
    X1_s=torch.zeros((Global.K_C+1)*2,Global.D_).float().cuda()

    X2_00 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_01 = torch.zeros(Global.K_C + 1).float().cuda()
    X2_11 = torch.zeros(Global.K_C + 1).float().cuda()

    X2_00_s = torch.zeros((Global.K_C + 1)*2).float().cuda()
    X2_01_s = torch.zeros((Global.K_C + 1)*2).float().cuda()
    X2_11_s = torch.zeros((Global.K_C + 1)*2).float().cuda()

    X_C_SIGMA = torch.zeros(Global.N, Global.neig_num, Global.D_).float().cuda()
    sum_buffer = torch.zeros(Global.N).float().cuda()
    clusters_LR=torch.zeros((Global.HEIGHT)*(Global.WIDTH),2).cuda().int()
    clusters_LR[:,0]=torch.arange(0,(Global.HEIGHT)*(Global.WIDTH)).int()
    XXT=torch.bmm(X[:, 0:2].unsqueeze(2),X[:, 0:2].unsqueeze(1)).reshape(-1,4)
    distances_buffer=torch.zeros(Global.N*Global.neig_num*Global.D_).float().cuda()
    r_ik_5=torch.zeros(Global.N,Global.neig_num).float().cuda()
    neig_buffer=torch.zeros(Global.N,Global.neig_num,Global.potts_area).float().cuda()
    sumP_buffer=torch.zeros(Global.N,Global.neig_num).float().cuda()
    X_C_buffer=torch.zeros(Global.N,Global.neig_num, Global.D_).float().cuda()
    X_C_SIGMA_buf=torch.zeros(Global.N,Global.neig_num,2).float().cuda()

    init=True
    it_merge=0
    it_split=0

    "Start Creating Buffers"

    SigmaXY_b = torch.zeros((Global.N + 1, Global.D_Inv)).cuda().float()
    SigmaXY_i_b = torch.zeros((Global.N + 1, Global.D_Inv)).cuda().float()

    SIGMAxylab_b = torch.zeros((Global.N + 1, Global.D_, Global.D_)).cuda().float()
    SIGMAxylab_b[:, 2, 2] = Global.int_scale
    SIGMAxylab_b[:, 3, 3] = Global.int_scale
    SIGMAxylab_b[:, 4, 4] = Global.int_scale
    if (Global.D_12):
        SIGMAxylab_b[:, 7, 7] = Global.int_scale
        SIGMAxylab_b[:, 8, 8] = Global.intl_scale
        SIGMAxylab_b[:, 9, 9] = Global.int_scale
        SIGMAxylab_b[:, 10, 10] = Global.opt_scale
        SIGMAxylab_b[:, 11, 11] = Global.opt_scale

    Nk_b = torch.zeros(Global.N + 1).float().cuda()
    X1_b = torch.zeros(Global.N + 1, Global.D_).float().cuda()

    X2_00_b = torch.zeros(Global.N + 1).float().cuda()
    X2_01_b = torch.zeros(Global.N + 1).float().cuda()
    X2_11_b = torch.zeros(Global.N + 1).float().cuda()

    sons_LL_b = torch.zeros(Global.N + 1, 4).float().cuda()
    X_sons_b = torch.zeros(Global.N + 1, 2).float().cuda()
    X_father_b = torch.zeros(Global.N + 1, 2).float().cuda()
    father_LL_b = torch.zeros(Global.N + 1, 4).float().cuda()

    m_v_sons_b = torch.zeros(Global.N + 1, 3).float().cuda()
    m_v_father_b = torch.zeros(Global.N + 1, 3).float().cuda()
    b_sons_b = torch.zeros(Global.N + 1, 3).float().cuda()
    b_father_b = torch.zeros(Global.N + 1, 3).float().cuda()

    merged_LL_b = torch.zeros(Global.N + 1, 4).float().cuda()
    X_merged_b = torch.zeros(Global.N + 1, 2).float().cuda()
    temp_b = torch.zeros((Global.K_C*10), 800).long().cuda()

    "End Creating Buffers"


    r_ik, pi = InitKmeansSP()
    #while(1):
    Global.K_C=Global.K_C_ORIGINAL
    SigmaXY = SigmaXY_b[0:Global.K_C + 1]
    SigmaXY_i = SigmaXY_i_b[0:Global.K_C + 1]
    temp=temp_b[0:Global.K_C+1]

    SIGMAxylab = SIGMAxylab_b[0:Global.K_C + 1]
    Nk = Nk_b[0:Global.K_C + 1]
    X1 = X1_b[0:Global.K_C + 1]

    X2_00 = X2_00_b[0:Global.K_C + 1]
    X2_01 = X2_01_b[0:Global.K_C + 1]
    X2_11 = X2_11_b[0:Global.K_C + 1]

    it_limit=310

    maxIt = 350

    fixIt_L = 305
    fixIt_H = 335
    it = 0
    split_turn=1
    print("CPU->GPU")

    argmax = torch.from_numpy(r_ik).cuda().unsqueeze(1).repeat(1,2)
    argmax_start=argmax.clone()

    argmax=argmax_start.clone()
    # pr=cProfile.Profile()
    #pr.enable()
    count_split=0
    count_merge=0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    while (it < maxIt and it<600):
        it += 1
        if (it % 20 == 0):
            print("It :", it)
        C1, logdet,SigmaXY, SigmaXY_i, Nk,SubData = EstimateSP_2Frames(X,loc, argmax, SigmaXY, SigmaXY_i, Nk, X1, X2_00, X2_01,X2_11,init,Nk_s,X1_s,X2_00_s,X2_01_s,X2_11_s,SigmaXY_s,SigmaXY_i_s,it,maxIt)#Nk_r,X1_r, X2_00_r, X2_01_r,X2_11_r,SigmaXY_r,SigmaXY_l,SigmaXY_i_r,SigmaXY_i_l)  # M-Step

        N_ = X.shape[0]
        pi_t = torch.div(torch.mul(Nk, (1 - Global.PI_0)), (N_ - Nk[0]))
        pi_t[0] = Global.PI_0_T


        if(it>fixIt_L and it%25==1 and it<fixIt_H):
            real_K_C=torch.sum((Nk>2).int())

        if (( (it) % 50 == 1 and (it<it_limit) and (it>120)) or ( (it) % 25 == 1 and it>fixIt_L and it<fixIt_H and real_K_C>Global.K_C_HIGH) ):
            if(it>it_limit):
                maxIt+=25
                fixIt_H+=25
                print("Fixing K_C ",real_K_C)
            padded_matrix = Global.Padding0(argmax[:,0].reshape(Global.HEIGHT,-1)).reshape(-1).cuda()

            pair = torch.zeros(Global.K_C + 1).int().cuda()
            left = torch.zeros(Global.K_C + 1,2).int().cuda()
            left[:,0]=torch.arange(0,Global.K_C+1)
            left[:,0]=Global.N_index[0:Global.K_C+1]


            if(it_merge%4==0):
                ind_left = torch.masked_select(Global.inside_padded, (
                            (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded - 1]) + (
                                padded_matrix[Global.inside_padded - 1] > 0)) == 2)
                left[padded_matrix[ind_left],1] = padded_matrix[ind_left - 1].int()
            if (it_merge%4 == 1):
                ind_left = torch.masked_select(Global.inside_padded, (
                        (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded + 1]) + (
                        padded_matrix[Global.inside_padded + 1] > 0)) == 2)
                left[padded_matrix[ind_left], 1] = padded_matrix[ind_left + 1].int()
            if (it_merge%4 == 2):
                ind_left = torch.masked_select(Global.inside_padded, (
                        (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded - (Global.WIDTH+2)]) + (
                        padded_matrix[Global.inside_padded - (Global.WIDTH+2)] > 0)) == 2)
                left[padded_matrix[ind_left], 1] = padded_matrix[ind_left - (Global.WIDTH+2)].int()
            if (it_merge%4 == 3):
                ind_left = torch.masked_select(Global.inside_padded, (
                        (padded_matrix[Global.inside_padded] != padded_matrix[Global.inside_padded + (Global.WIDTH+2)]) + (
                        padded_matrix[Global.inside_padded + (Global.WIDTH+2)] > 0)) == 2)
                left[padded_matrix[ind_left], 1] = padded_matrix[ind_left + (Global.WIDTH+2)].int()

            ind_left,_=torch.sort(ind_left)
            sorted, indices = torch.sort(padded_matrix[ind_left])
            Nk.zero_()
            Nk.index_add_(0, sorted, Global.ones[0:sorted.shape[0]])
            temp = temp_b[0:Global.K_C + 1].zero_()
            tempNk=torch.cumsum(Nk,0).long()

            temp_ind=ind_left[indices.long()].long()


            if(it_merge%4==0):
                temp[padded_matrix[temp_ind].long(), tempNk[padded_matrix[temp_ind].long()].long() - Global.N_index[
                                                                                                     0:ind_left.shape[
                                                                                                         0]].long()] = \
                padded_matrix[temp_ind - 1].long()

            if (it_merge%4 == 1):
                temp[padded_matrix[temp_ind].long(), tempNk[padded_matrix[temp_ind].long()].long() - Global.N_index[
                                                                                                     0:ind_left.shape[
                                                                                                         0]].long()] = \
                padded_matrix[temp_ind + 1].long()

            if (it_merge%4 == 2):
                temp[padded_matrix[temp_ind].long(), tempNk[padded_matrix[temp_ind].long()].long() - Global.N_index[
                                                                                                     0:ind_left.shape[
                                                                                                         0]].long()] = \
                padded_matrix[temp_ind - (Global.WIDTH+2)].long()

            if (it_merge%4 == 3):
                temp[padded_matrix[temp_ind].long(), tempNk[padded_matrix[temp_ind].long()].long() - Global.N_index[
                                                                                                     0:ind_left.shape[
                                                                                                         0]].long()] = \
                padded_matrix[temp_ind + (Global.WIDTH+2)].long()

            #temp[padded_matrix[temp_ind].long(),tempNk[padded_matrix[temp_ind].long()].long()-Global.N_index[0:ind_left.shape[0]].long()]=padded_matrix[temp_ind-1].long()
            left[:,1]=torch.max(temp,dim=1)[0]
            # temp_ind,vals=torch.unique((padded_matrix[ind_left-1]),return_inverse=True)
            it_merge=it_merge+1


            # left[:,1].zero_()
            #
            # for i in range(0,ind_left.shape[0]):
            #     if(left[padded_matrix[ind_left[i]],1]==0):
            #         left[padded_matrix[ind_left[i]], 1]=padded_matrix[ind_left[i]+1]

            for i in range(0, Global.K_C + 1):

                val = left[i, 1]
                if ((val > 0 )and (val!=i)):
                    if ((pair[i] == 0) and (pair[val] == 0)):
                        if (val < i):
                            pair[i] = val
                            pair[val] = val
                        else:
                            pair[val] = i
                            pair[i] = i

            left[:,1]=pair


            # Nk.zero_()
            # Nk.index_add_(0, argmax[:, 0], Global.ones)
            # Nk = Nk + 0.0000000001
            # Nk_merged=torch.add(Nk,Nk[left[:,1].long()])
            #
            #
            #

            # merged_LL = merged_LL_b[0:Global.K_C+1].zero_()
            # X_merged= X_merged_b[0:Global.K_C+1].zero_()
            # X_father = X_father_b[0:Global.K_C + 1].zero_()
            # father_LL = father_LL_b[0:Global.K_C + 1].zero_()
            #
            # X_merged.index_add_(0,left[argmax[:,0],1].long(),X[:,0:2])
            # X_merged=X_merged[left[:,1].long()]
            # X_father.index_add_(0,argmax[:,0],X[:,0:2])
            #
            # merged_LL[:,0]= -torch.pow(X_merged[:,0],2)
            # merged_LL[:,1]= -torch.mul(X_merged[:,0],X_merged[:,1])
            # merged_LL[:,2]= -merged_LL[:,1]
            # merged_LL[:,3]= -torch.pow(X_merged[:,1],2)
            #
            # father_LL[:, 0] = -torch.pow(X_father[:,0], 2)
            # father_LL[:, 1] = -torch.mul(X_father[:,0], X_father[:,1])
            # father_LL[:, 2] = -father_LL[:, 1]
            # father_LL[:, 3] = -torch.pow(X_father[:,1], 2)


            # a_prior_merged = torch.mul(torch.pow(2, Global.split_lvl[0:Nk_merged.shape[0]] + 1), Global.A_prior)
            # psi_prior_merged = torch.mul(torch.pow(a_prior_merged, 2).unsqueeze(1),torch.eye(2).reshape(-1, 4).cuda())
            # ni_prior_merged = (Global.C_prior * a_prior_merged) - 3
            #
            # merged_LL.index_add_(0, left[argmax[:,0],1].long(), XXT)
            # merged_LL=merged_LL[left[:,1].long()]
            # father_LL.index_add_(0,argmax[:,0],XXT)
            # ni_merged=torch.add(ni_prior_merged,Nk_merged)[0:merged_LL.shape[0]]
            # ni_father=torch.add(Global.ni_prior,Nk)[0:father_LL.shape[0]]
            # psi_merged=torch.add(merged_LL.reshape(-1,4),psi_prior_merged)[0:ni_merged.shape[0]]
            # psi_father=torch.add(father_LL.reshape(-1,4),Global.psi_prior)[0:ni_father.shape[0]]
            #
            # ni_merged[(ni_merged <= 1).nonzero()] = 2.00000001
            # ni_father[(ni_father <= 1).nonzero()] = 2.00000001
            #
            # gamma_merged = torch.mvlgamma((ni_merged / 2), 2)
            # gamma_father = torch.mvlgamma((ni_father / 2), 2)
            #
            #
            #
            # det_psi_merged=0.00000001+torch.add(torch.mul(psi_merged[:, 0], psi_merged[:, 3]),-torch.mul(psi_merged[:, 1], psi_merged[:, 2]))
            # det_psi_father=0.00000001+torch.add(torch.mul(psi_father[:, 0], psi_father[:, 3]),-torch.mul(psi_father[:, 1], psi_father[:, 2]))
            # det_psi_merged[(det_psi_merged <= 0).nonzero()] = 0.00000001
            # det_psi_father[(det_psi_father <= 0).nonzero()] = 0.00000001
            #
            # det_psi_prior_merged=0.00000001+torch.add(torch.mul(psi_prior_merged[:, 0], psi_prior_merged[:, 3]),-torch.mul(psi_prior_merged[:, 1], psi_prior_merged[:, 2]))
            # det_psi_prior_father=0.00000001+torch.add(torch.mul(Global.psi_prior[:, 0], Global.psi_prior[:, 3]),-torch.mul(Global.psi_prior[:, 1], Global.psi_prior[:, 2]))
            # det_psi_prior_merged[(det_psi_prior_merged <= 0).nonzero()] = 0.00000001
            # det_psi_prior_father[(det_psi_prior_father <= 0).nonzero()] = 0.00000001
            #
            #
            # ni_prior_merged[(ni_prior_merged <= 1).nonzero()] = 2.00000001
            # Global.ni_prior[(Global.ni_prior <= 1).nonzero()] = 2.00000001
            # gamma_prior_merged=torch.mvlgamma(ni_prior_merged/2,2)
            # gamma_prior_father=torch.mvlgamma(Global.ni_prior /2,2)
            #
            # ll_merged= -(torch.mul(torch.log((Global.PI)),(Nk_merged)))+ \
            #          torch.add(gamma_merged,-gamma_prior_merged) + \
            #          torch.mul(torch.log(det_psi_prior_merged), (ni_prior_merged * 0.5)) - \
            #          torch.mul(torch.log(det_psi_merged),(ni_merged * 0.5))+\
            #          torch.log(Nk_merged[0:merged_LL.shape[0]])
            #
            # ll_father= -(torch.mul(torch.log((Global.PI)),(Nk)))+ \
            #            torch.add(gamma_father,-gamma_prior_father) + \
            #            torch.mul(torch.log((det_psi_father)), Global.ni_prior * 0.5) - \
            #            torch.mul(torch.log(det_psi_father),ni_father * 0.5) +\
            #            torch.log(Nk[0:father_LL.shape[0]])
            #
            # ll_merged_min=torch.min(ll_merged[1:ll_merged.shape[0]].masked_select(1-torch.isinf(ll_merged[1:ll_merged.shape[0]])-torch.isnan(ll_merged[1:ll_merged.shape[0]])))
            # ll_merged_max=torch.max(ll_merged[1:ll_merged.shape[0]].masked_select(1-torch.isinf(ll_merged[1:ll_merged.shape[0]])-torch.isnan(ll_merged[1:ll_merged.shape[0]])))
            # ll_father_min=torch.min(ll_father[1:ll_father.shape[0]].masked_select(1-torch.isinf(ll_father[1:ll_father.shape[0]])-torch.isnan(ll_father[1:ll_father.shape[0]])))
            # ll_father_max=torch.max(ll_father[1:ll_father.shape[0]].masked_select(1-torch.isinf(ll_father[1:ll_father.shape[0]])-torch.isnan(ll_father[1:ll_father.shape[0]])))
            #
            # ll_merged_min = torch.min(ll_merged_min, ll_father_min)
            # ll_merged_max = torch.max(ll_merged_max, ll_father_max)
            #
            # ll_merged=torch.div(torch.add(ll_merged,-ll_merged_min),(ll_merged_max-ll_merged_min))*(-1000)+0.1
            # ll_father = torch.div(torch.add(ll_father, -ll_merged_min), (ll_merged_max - ll_merged_min))*(-1000) + 0.1
            # #

            Nk.zero_()
            Nk.index_add_(0, argmax[:, 0], Global.ones)
            Nk = Nk + 0.0000000001

            Nk_merged=torch.add(Nk,Nk[left[:,1].long()])
            # Nk_merged[left[:,1].long()]=Nk_merged[left[:,0].long()]
            alpha=torch.Tensor([float(1000000)]).cuda()
            beta=torch.Tensor([Global.int_scale*alpha+Global.int_scale]).cuda()
            v_father = Nk
            v_merged = Nk_merged


            m_v_father = m_v_father_b[0:Nk.shape[0]].zero_()
            b_father = b_father_b[0:Nk.shape[0]].zero_()

            m_v_father.index_add_(0, argmax[:, 0], X[:, 2:5])
            m_v_merged = torch.add(m_v_father, m_v_father[left[:, 1].long()])


            m_merged = torch.div(m_v_merged, v_merged.unsqueeze(1))
            m_father = torch.div(m_v_father, v_father.unsqueeze(1))
            a_father = torch.add(Nk / 2, alpha).unsqueeze(1)
            a_merged = torch.add(Nk_merged / 2, alpha).unsqueeze(1)
            b_father.index_add_(0, argmax[:, 0], torch.pow(X[:, 2:5], 2))
            b_merged=torch.add(b_father,b_father[left[:,1].long()])
            b_father=b_father/2
            b_merged=b_merged/2
            b_father.add_(torch.add(beta, -torch.mul(torch.pow(m_father, 2), v_father.unsqueeze(1)) / 2))
            b_merged.add_(torch.add(beta, -torch.mul(torch.pow(m_merged, 2), v_merged.unsqueeze(1)) / 2))


            gamma_2_merged = torch.mvlgamma(a_merged,1)
            gamma_2_father = torch.mvlgamma(a_father, 1)


            ll_2_merged=0.5*torch.log(v_merged).unsqueeze(1)+ \
                        (a_merged*torch.log(b_merged))+\
                        gamma_2_merged- \
                        ((torch.mul(torch.log(Global.PI),Nk_merged/2))+(0.301*Nk_merged)).unsqueeze(1)


            ll_2_father=0.5*torch.log(v_father).unsqueeze(1)+ \
                        (a_father*torch.log(b_father))+\
                        gamma_2_father- \
                        ((torch.mul(torch.log(Global.PI),Nk/2))+(0.301*Nk)).unsqueeze(1)


            ll_2_father = torch.sum(ll_2_father, 1)[0:ll_2_father.shape[0]]
            ll_2_merged = torch.sum(ll_2_merged, 1)[0:ll_2_merged.shape[0]]



            ll_merged_min = torch.min(ll_2_merged[1:ll_2_merged.shape[0]].masked_select(
                1 - torch.isnan(ll_2_merged[1:ll_2_merged.shape[0]]) - torch.isinf(ll_2_merged[1:ll_2_merged.shape[0]])))
            ll_merged_max = torch.max(ll_2_merged[1:ll_2_merged.shape[0]].masked_select(
                1 - torch.isnan(ll_2_merged[1:ll_2_merged.shape[0]]) - torch.isinf(ll_2_merged[1:ll_2_merged.shape[0]])))
            ll_father_min = torch.min(
                ll_2_father.masked_select(1 - torch.isnan(ll_2_father) - torch.isinf(ll_2_father)))
            ll_father_max = torch.max(
                ll_2_father.masked_select(1 - torch.isnan(ll_2_father) - torch.isinf(ll_2_father)))

            ll_merged_min=torch.min(ll_merged_min,ll_father_min)
            ll_merged_max=torch.max(ll_merged_max,ll_father_max)
            # ll_2_merged = torch.div(torch.add(ll_2_merged, -ll_merged_min), (ll_merged_max - ll_merged_min)) * (-1000) + 0.1
            # ll_2_father = torch.div(torch.add(ll_2_father, -ll_father_min), (ll_father_max - ll_father_min)) * (-1000) + 0.1


            ll_2_merged = torch.div(torch.add(ll_2_merged, -ll_merged_min), (ll_merged_max - ll_merged_min)) * (-1000) + 0.1
            ll_2_father = torch.div(torch.add(ll_2_father, -ll_merged_min), (ll_merged_max - ll_merged_min))*(-1000) + 0.1




            gamma_alpha_2=torch.mvlgamma(torch.Tensor([Global.ALPHA_MS2/2]).cuda(),1)
            gamma_alpha=torch.mvlgamma(torch.Tensor([Global.ALPHA_MS2]).cuda(),1)

            gamma_father=torch.mvlgamma(Nk,1)
            gamma_add_father=torch.mvlgamma(Nk_merged,1)
            gamma_alpha_father=torch.mvlgamma(Nk+Global.ALPHA_MS2/2,1)
            gamma_add_alpha_merged = torch.mvlgamma(Nk_merged + Global.ALPHA_MS2, 1)

            # ll_2_merged.add_(ll_merged)
            # ll_2_father.add_(ll_father)
            #
            # prob= (-2) + \
            #       (gamma_add_father) -\
            #       (Global.ALPHA_MS2+gamma_father[left[:,0].long()]+gamma_father[left[:,1].long()])+\
            #       (ll_2_merged[left[:,0].long()]- ll_2_father[left[:,0].long()] -ll_2_father[[left[:,1].long()]])+\
            #       (gamma_alpha-gamma_add_alpha_merged)+\
            #       (gamma_alpha_father[left[:,0].long()]+gamma_alpha_father[left[:,1].long()])-\
            #       (gamma_alpha_2+gamma_alpha_2)

            # prob= -Global.ALPHA_MS2 +\
            #       (ll_2_merged[left[:, 0].long()] - ll_2_father[left[:, 0].long()] - ll_2_father[[left[:, 1].long()]])

            prob = -Global.LOG_ALPHA_MS2+gamma_alpha-2*gamma_alpha_2 +\
                   gamma_add_father-gamma_add_alpha_merged+ \
                   gamma_alpha_father[left[:, 0].long()]-gamma_father[left[:, 0].long()]+ \
                   gamma_alpha_father[left[:, 1].long()] - gamma_father[left[:, 1].long()] - 2 + \
                   ll_2_merged[left[:, 0].long()] - ll_2_father[left[:, 0].long()] - ll_2_father[[left[:, 1].long()]]

            # prob=torch.where((Nk<6),torch.Tensor([float("inf")]).cuda(),prob)
            prob=torch.where(((left[:,0]==left[:,1])+(left[:,1]==0))>0,-torch.Tensor([float("inf")]).cuda(),prob)

            idx_rand=torch.where(torch.exp(prob) > 1.0, Global.N_index[0:prob.shape[0]].long(),Global.zeros[0:prob.shape[0]].long()).nonzero()[:, 0]

            pair[left[:,1].long()]=left[left[:,0].long()][:,0]

            #pair_temp = torch.arange(0, Global.K_C + 1).cuda().int()
            left[:,1]=Global.N_index[0:Global.K_C+1]
            left[idx_rand.long(),1]=pair[idx_rand.long()]

            argmax[:,0] = left[argmax[:,0],1]



            print("Idx Merge Size: ",idx_rand.shape[0])
            # if(idx_rand.shape[0]==0):
            #     count_no_merge=count_no_merge+1
            Global.split_lvl.index_add_(0,left[idx_rand,1].long(),1*Global.ones[0:idx_rand.shape[0]])
            Global.split_lvl.index_add_(0,idx_rand,1*Global.ones[0:idx_rand.shape[0]])


        if (Global.HARD_EM == True or init==True):
            prev_r_ik_max = argmax[:,0].clone()


        c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)
        c_idx_9 = prev_r_ik_max.view(-1).index_select(0, Global.c_idx_9)
        c_idx_25 = prev_r_ik_max.view(-1).index_select(0, Global.c_idx_25)


        prev_r_ik_max = prev_r_ik_max.view((Global.HEIGHT, Global.WIDTH))
        c1_vals = C1.index_select(0, c_idx).view(-1)
        pi_vals = pi_t.index_select(0, c_idx).view(-1)
        SigmaXY_vals = SigmaXY_i.index_select(0, c_idx).view(-1)
        logdet_vals = logdet.index_select(0, c_idx).view(-1)
        c1_temp.scatter_(0, range1, c1_vals)
        pi_temp.scatter_(0, range3, pi_vals)
        logdet_temp.scatter_(0, range3, logdet_vals)
        SigmaXY_temp.scatter_(0, range4, SigmaXY_vals)

        if(init):
            init=False
            idx_rand=torch.arange(1,Global.K_C+1).cuda()
            argmax[:,1],split,clusters_LR=my_connectivity.Split(prev_r_ik_max,prev_r_ik_max, C1, c1_temp,idx_rand,clusters_LR,it_split)




        FindClosestClusterSP_2Frames(X, logdet_temp, c1_temp, pi_temp,  SigmaXY_temp.reshape(-1, Global.neig_num, Global.D_Inv), X_C_SIGMA, sum_buffer,c_idx,c_idx_9,c_idx_25,distances_buffer,r_ik_5,neig_buffer,sumP_buffer,X_C_buffer,X_C_SIGMA_buf) #TODO: Check  r_ik_5 pointer
        r_ik_5 = r_ik_5.view(-1, Global.neig_num)



        if(Global.HARD_EM==True):
            argmax[:,0] = my_connectivity.Change_pixel(prev_r_ik_max, r_ik_5, index2_buffer, r_ikNew_buffer, it % 4, c_idx, _,range_conn)#,split_prev_r_ik_max,c_idx_split,r_ik_5_s)
        if (( (it+25)%50   == 1  and it>120 and it<it_limit ) or (it%25==1 and it>fixIt_L and it<fixIt_H and real_K_C<Global.K_C_LOW)):
            if(it>it_limit):
                maxIt+=25
                fixIt_H+=25
                print("Fixing K_C ",real_K_C)
                if(real_K_C==217):
                    aa=5
            argmax[:, 1],sub_clusters,clusters_LR=my_connectivity.Split(argmax[:,0].reshape(Global.HEIGHT,-1),argmax[:,0].reshape(Global.HEIGHT,-1), C1, c1_temp,Global.N_index[1:Global.K_C+1],clusters_LR,it_split)
            it_split=it_split+1
            K_C_Split=torch.max(argmax[:,1])+1
            if(Nk.shape[0]>K_C_Split):
                K_C_Split=Nk.shape[0]
            Nk_s = torch.zeros(K_C_Split).float().cuda()
            Nk.zero_()
            Global.C_prior = 11 + (torch.mul(0, -Global.split_lvl[0:Nk_s.shape[0]]))
            # Global.C_prior = 1550
            a_prior_sons = Nk_s
            Global.psi_prior_sons = torch.mul(torch.pow(a_prior_sons, 2).unsqueeze(1), torch.eye(2).reshape(-1, 4).cuda())
            Global.ni_prior_sons = (Global.C_prior * a_prior_sons) - 3
            Nk.index_add_(0, argmax[:, 0], Global.ones)
            Nk = Nk + 0.0000000001
            Nk_s.index_add_(0, argmax[:, 1], Global.ones)
            Nk_s = Nk_s + 0.0000000001



            sons_LL=sons_LL_b[0:Nk_s.shape[0]].zero_()
            X_sons=X_sons_b[0:Nk_s.shape[0]].zero_()
            X_father=X_father_b[0:Global.K_C+1].zero_()
            father_LL=father_LL_b[0:Global.K_C+1].zero_()


            X_sons.index_add_(0,argmax[:,1],X[:,0:2])
            X_father.index_add_(0,argmax[:,0],X[:,0:2])
            sons_LL[:,0]= -torch.pow(X_sons[:,0],2)
            sons_LL[:,1]= -torch.mul(X_sons[:,0],X_sons[:,1])
            sons_LL[:,2]= -sons_LL[:,1]
            sons_LL[:,3]= -torch.pow(X_sons[:,1],2)

            father_LL[:, 0] = -torch.pow(X_father[:,0], 2)
            father_LL[:, 1] = -torch.mul(X_father[:,0], X_father[:,1])
            father_LL[:, 2] = -father_LL[:, 1]
            father_LL[:, 3] = -torch.pow(X_father[:,1], 2)


            sons_LL.index_add_(0, argmax[:,1], XXT)
            father_LL.index_add_(0,argmax[:,0],XXT)

            ni_sons=torch.add(Global.ni_prior_sons,Nk_s)[0:sons_LL.shape[0]]
            ni_father=torch.add(Global.ni_prior,Nk)[0:father_LL.shape[0]]
            psi_sons=torch.add(sons_LL.reshape(-1,4),Global.psi_prior_sons)[0:ni_sons.shape[0]]
            psi_father=torch.add(father_LL.reshape(-1,4),Global.psi_prior)[0:ni_father.shape[0]]
            ni_sons[(ni_sons <= 1).nonzero()] = 2.00000001
            ni_father[(ni_father <= 1).nonzero()] = 2.00000001

            gamma_sons=torch.mvlgamma((ni_sons/2),2)
            gamma_father=torch.mvlgamma((ni_father/2),2)
            det_psi_sons=0.00000001+torch.add(torch.mul(psi_sons[:, 0], psi_sons[:, 3]),-torch.mul(psi_sons[:, 1], psi_sons[:, 2]))
            det_psi_father=0.00000001+torch.add(torch.mul(psi_father[:, 0], psi_father[:, 3]),-torch.mul(psi_father[:, 1], psi_father[:, 2]))
            det_psi_sons[(det_psi_sons <= 0).nonzero()] = 0.00000001
            det_psi_father[(det_psi_father <= 0).nonzero()] = 0.00000001

            det_psi_prior_sons=0.00000001+torch.add(torch.mul(Global.psi_prior_sons[:, 0], Global.psi_prior_sons[:, 3]),-torch.mul(Global.psi_prior_sons[:, 1], Global.psi_prior_sons[:, 2]))
            det_psi_prior_father=0.00000001+torch.add(torch.mul(Global.psi_prior[:, 0], Global.psi_prior[:, 3]),-torch.mul(Global.psi_prior[:, 1], Global.psi_prior[:, 2]))
            det_psi_prior_sons[(det_psi_prior_sons <= 0).nonzero()] = 0.00000001
            det_psi_prior_father[(det_psi_prior_father <= 0).nonzero()] = 0.00000001

            Global.ni_prior_sons[(Global.ni_prior_sons <= 1).nonzero()] = 2.00000001
            Global.ni_prior[(Global.ni_prior <= 1).nonzero()] = 2.00000001
            gamma_prior_sons=torch.mvlgamma((Global.ni_prior_sons / 2),2)
            gamma_prior_father=torch.mvlgamma((Global.ni_prior / 2),2)

            ll_sons= -(torch.mul(torch.log((Global.PI)),(Nk_s)))+ \
                     torch.add(gamma_sons,-gamma_prior_sons) + \
                     torch.mul(torch.log(det_psi_prior_sons), (Global.ni_prior_sons * 0.5)) - \
                     torch.mul(torch.log(det_psi_sons),(ni_sons * 0.5))+\
                     torch.log(Nk_s[0:sons_LL.shape[0]])

            ll_father= -(torch.mul(torch.log((Global.PI)),(Nk)))+ \
                       torch.add(gamma_father,-gamma_prior_father) + \
                       torch.mul(torch.log((det_psi_father)), Global.ni_prior * 0.5) - \
                       torch.mul(torch.log(det_psi_father),ni_father * 0.5) +\
                       torch.log(Nk[0:father_LL.shape[0]])

            ll_sons_min=torch.min(ll_sons[1:ll_sons.shape[0]].masked_select(1-torch.isinf(ll_sons[1:ll_sons.shape[0]])-torch.isnan(ll_sons[1:ll_sons.shape[0]])))
            ll_sons_max=torch.max(ll_sons[1:ll_sons.shape[0]].masked_select(1-torch.isinf(ll_sons[1:ll_sons.shape[0]])-torch.isnan(ll_sons[1:ll_sons.shape[0]])))
            ll_father_min=torch.min(ll_father[1:ll_father.shape[0]].masked_select(1-torch.isinf(ll_father[1:ll_father.shape[0]])-torch.isnan(ll_father[1:ll_father.shape[0]])))
            ll_father_max=torch.max(ll_father[1:ll_father.shape[0]].masked_select(1-torch.isinf(ll_father[1:ll_father.shape[0]])-torch.isnan(ll_father[1:ll_father.shape[0]])))

            ll_sons_min=torch.min(ll_sons_min,ll_father_min)
            ll_sons_max=torch.max(ll_sons_max,ll_father_max)


            ll_sons=torch.div(torch.add(ll_sons,-ll_sons_min),(ll_sons_max-ll_sons_min))*(-1000)+0.1
            ll_father=torch.div(torch.add(ll_father,-ll_sons_min),(ll_sons_max-ll_sons_min))*(-1000)+0.1




            alpha=torch.Tensor([float(1000000)]).cuda()
            beta=torch.Tensor([Global.int_scale*alpha+Global.int_scale]).cuda()
            Nk.zero_()
            Nk_s.zero_()
            Nk.index_add_(0, argmax[:, 0], Global.ones)
            Nk = Nk + 0.0000000001
            Nk_s.index_add_(0, argmax[:, 1], Global.ones)
            Nk_s = Nk_s + 0.0000000001
            v_father=Nk
            v_sons=Nk_s




            m_v_sons=m_v_sons_b[0:Nk_s.shape[0]].zero_()
            m_v_father=m_v_father_b[0:Nk.shape[0]].zero_()
            b_sons = b_sons_b[0:Nk_s.shape[0]].zero_()
            b_father = b_father_b[0:Nk.shape[0]].zero_()
            m_v_sons.index_add_(0, argmax[:, 1], X[:,2:5])
            m_v_father.index_add_(0, argmax[:, 0], X[:,2:5])
            m_sons=torch.div(m_v_sons,v_sons.unsqueeze(1))
            m_father=torch.div(m_v_father,v_father.unsqueeze(1))
            a_sons=torch.add(Nk_s/2,alpha).unsqueeze(1)
            a_father=torch.add(Nk/2,alpha).unsqueeze(1)
            b_sons.index_add_(0, argmax[:, 1], torch.pow(X[:, 2:5],2))
            b_father.index_add_(0, argmax[:, 0],torch.pow(X[:, 2:5],2))
            b_sons=b_sons/2
            b_father=b_father/2
            b_sons.add_(torch.add(beta,-torch.mul(torch.pow(m_sons,2),v_sons.unsqueeze(1))/2))
            b_father.add_(torch.add(beta,-torch.mul(torch.pow(m_father,2),v_father.unsqueeze(1))/2))

            gamma_2_sons=torch.mvlgamma(a_sons,1)
            gamma_2_father=torch.mvlgamma(a_father,1)

            ll_2_sons=(0.5*torch.log(v_sons).unsqueeze(1))+\
                      (torch.log(beta)*alpha)-\
                      (a_sons*torch.log(b_sons))+\
                      gamma_2_sons-\
                      ((torch.mul(torch.log(Global.PI),Nk_s/2))+(0.301*Nk_s)).unsqueeze(1)
            ll_2_father=0.5*torch.log(v_father).unsqueeze(1)+ \
                        (a_father*torch.log(b_father))+\
                        gamma_2_father- \
                        ((torch.mul(torch.log(Global.PI),Nk/2))+(0.301*Nk)).unsqueeze(1)

            ll_2_sons=torch.sum(ll_2_sons,1)[0:ll_sons.shape[0]]
            ll_2_father = torch.sum(ll_2_father, 1)[0:ll_father.shape[0]]

            ll_sons_min = torch.min(ll_2_sons[1:ll_2_sons.shape[0]].masked_select(1 - torch.isnan(ll_2_sons[1:ll_2_sons.shape[0]])-torch.isinf(ll_2_sons[1:ll_2_sons.shape[0]])))
            ll_sons_max = torch.max(ll_2_sons[1:ll_2_sons.shape[0]].masked_select(1 - torch.isnan(ll_2_sons[1:ll_2_sons.shape[0]])-torch.isinf(ll_2_sons[1:ll_2_sons.shape[0]])))
            ll_father_min = torch.min(ll_2_father.masked_select(1 - torch.isnan(ll_2_father)-torch.isinf(ll_2_father)))
            ll_father_max = torch.max(ll_2_father.masked_select(1 - torch.isnan(ll_2_father)-torch.isinf(ll_2_father)))



            ll_2_sons = torch.div(torch.add(ll_2_sons, -ll_sons_min), (ll_sons_max - ll_sons_min))*(-1000) + 0.1
            ll_2_father = torch.div(torch.add(ll_2_father, -ll_father_min), (ll_father_max - ll_father_min))*(-1000) + 0.1

            ll_sons.add_(ll_2_sons)
            ll_father.add_(ll_2_father)



            gamma_1_sons=torch.mvlgamma(Nk_s,1)
            gamma_1_father=torch.mvlgamma(Nk,1)
            ll_sons=torch.where(Nk_s[0:ll_sons.shape[0]]<35,Global.zeros[0:ll_sons.shape[0]]- torch.Tensor([float("inf")]).cuda(),ll_sons)
            ind_sons=clusters_LR[0:gamma_1_sons.shape[0]].long()
            ind_sons[ind_sons>ll_sons.shape[0]-1]=0 #TODO:Check if relevant
            prob=(Global.ALPHA_MS)+\
                 ((ll_sons[ind_sons[:,0]]+\
                   gamma_1_sons[ind_sons[:,0]]+\
                   ll_sons[ind_sons[:,1]]+\
                   gamma_1_sons[ind_sons[:,1]])[0:gamma_1_father.shape[0]-1]-
                  ((gamma_1_father)+ll_father)[0:gamma_1_father.shape[0]-1])


            idx_rand=torch.where(torch.exp(prob) > 1.0, Global.N_index[0:prob.shape[0]].long(),Global.zeros[0:prob.shape[0]].long()).nonzero()[:, 0]
            print("Idx Split Size: ",idx_rand.shape[0])
            # if(idx_rand.shape[0]==0):
            #     count_no_split=count_no_split+1



            left = torch.zeros(Global.K_C + 1, 2).int().cuda()
            left[:, 0] = Global.N_index[0:Global.K_C + 1]
            left[idx_rand,1]=1
            pixels_to_change = left[argmax[:,0],1]

            original=torch.where(pixels_to_change==1,argmax[:, 1],argmax[:,0])

            left2 = torch.zeros(Global.K_C*2 + 1, 2).int().cuda()
            left2[:, 0] = Global.N_index[0:Global.K_C*2 + 1]
            left2[:, 1] = Global.N_index[0:Global.K_C*2 + 1]
            left2[clusters_LR[idx_rand,1].long(),1]=Global.N_index[0:idx_rand.shape[0]].int()+Global.K_C+1
            original=left2[original,1]
            Global.split_lvl.index_add_(0, idx_rand, -Global.ones[0:idx_rand.shape[0]])

            Global.split_lvl.index_add_(0,  left2[clusters_LR[idx_rand,1].long(),1].long(), -Global.ones[0:idx_rand.shape[0]])


            Global.K_C=torch.max(original.reshape(-1)).int()


            SigmaXY = SigmaXY_b[0:Global.K_C + 1]
            SigmaXY_i = SigmaXY_i_b[0:Global.K_C + 1]

            SIGMAxylab = SIGMAxylab_b[0:Global.K_C + 1]
            Nk = Nk_b[0:Global.K_C + 1]
            X1 = X1_b[0:Global.K_C + 1]

            X2_00 = X2_00_b[0:Global.K_C + 1]
            X2_01 = X2_01_b[0:Global.K_C + 1]
            X2_11 = X2_11_b[0:Global.K_C + 1]

            prev_r_ik_max = argmax[:, 0].clone()
            argmax[:,0]=original.reshape(-1)

        else:
            prev_r_ik_max = r_ik_5.argmax(1)
            prev_r_ik_max=torch.take(c_idx,torch.add(prev_r_ik_max,range_conn))

    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    # pr.disable()
    # pr.print_stats(sort="cumtime")

    #Fixing Single Points Clusters"

    Nk.zero_()
    Nk.index_add_(0, argmax[:, 0], Global.ones)
    c_idx = prev_r_ik_max.view(-1).index_select(0, Global.c_idx)
    c_idx=c_idx.reshape(-1, Global.neig_num)[:,1]
    argmax[:,0]=torch.where(Nk[argmax[:,0]]==1,c_idx,argmax[:,0])

    #Fixing Single Points Clusters"

    print("GPU->CPU")

    #argmax=split.reshape(-1).unsqueeze(1)

    r_ik = argmax[:,0].cpu().numpy()
    SigmaXY_cpu = SigmaXY.cpu().numpy()[:, 0:4]
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH)).astype(int)
    C1_cpu = C1.cpu().numpy()[:, 0:2]
    painted=np.zeros(Global.K_C+1)
    framePointsNew = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, 3))

    mean_value = np.zeros((Global.K_C + 1, 3))
    mean_value2=np.array([0,0,255])
    Global.split_lvl=Global.split_lvl.cpu().numpy()[0:Global.K_C+1]
    Global.split_lvl[Global.split_lvl!=1]*=1
    mean_value_temp=np.array(Global.colors)[1-(Global.split_lvl).astype(np.int)]
    mean_value2=np.zeros((Global.K_C+1,3))
    for i in range(0,Global.K_C+1):
        mean_value2[i]=np.array(my_help.hex2rgb(mean_value_temp[i]))
        mean_value2[i] = np.array([0, 0, 255])
        a=3

    framePointsNew2 = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, 3))
    framePointsNew3 = np.zeros((Global.HEIGHT+2, Global.WIDTH+2, 3)).astype(np.int)
    framePointsNew3[1:Global.HEIGHT+1,1:Global.WIDTH+1]=Global.frame0.astype(np.int)
    # Trueflow = np.expand_dims(C1_cpu[:, 10:12], 1)
    # Trueflow = Trueflow.reshape(-1, 1, 2)
    #flow = my_help.computeImg(Trueflow)

    #
    # hsv = np.zeros((Global.K_C+1,1,3), dtype=np.uint8)
    # hsv[..., 1] = 255
    #
    # mag, ang = cv2.cartToPolar(C2[..., 0], C2[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)
        # mean_value2[i, 0] = flow[i, :, 0]
        # mean_value2[i, 1] = flow[i, :, 1]
        # mean_value2[i, 2] = flow[i, :, 2]
        #
    # np.save('Frame_'+str(Global.FrameN)  + '_mean_RGB', mean_value)
    # np.save('Frame_'+str(Global.FrameN)  + '_mean_XY', C1_cpu)
    # np.save('Frame_'+str(Global.FrameN)  + '_SigmaXY', SigmaXY_cpu()
    # np.save('Frame_'+str(Global.FrameN)  + '_Clusters', r_ik2)

    padded=np.pad(r_ik2, 1, pad_with, padder=0)
    # framePointsNew2=np.pad(framePointsNew2, 1, pad_with, padder=0)
    # framePointsNew=np.pad(framePointsNew, 1, pad_with, padder=0)



    for i in range(padded.shape[0]):
        for j in range(padded.shape[1]):
            if(padded[i,j]!=0):
                framePointsNew[i, j] = mean_value[padded[i, j]]
                framePointsNew2[i, j] = mean_value[padded[i, j]]
                if(((padded[i+1,j]>0)and(padded[i,j]!=padded[i+1,j])) or ((padded[i,j-1]>0)and(padded[i,j]!=padded[i,j-1])) or ((padded[i,j+1]>0)and(padded[i,j]!=padded[i,j+1])) or ((padded[i-1,j]>0)and(padded[i,j]!=padded[i-1,j]))):
                    framePointsNew2[i,j]=mean_value2[padded[i,j]]
                    framePointsNew3[i,j]=mean_value2[padded[i,j]]

    fig = plt.figure()
    painted=np.zeros(Global.K_C+1)
    count=0
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if(painted[r_ik2[i,j]]==0):
                count=count+1
                painted[r_ik2[i,j]]=1

    framePointsNew=framePointsNew[1:Global.HEIGHT+1,1:Global.WIDTH+1]
    framePointsNew2=framePointsNew2[1:Global.HEIGHT+1,1:Global.WIDTH+1]
    framePointsNew3=framePointsNew3[1:Global.HEIGHT+1,1:Global.WIDTH+1]
    if(Global.Sintel):
        cv2.imwrite('SintelOutput/'+Global.SintelSave, r_ik2)
        Global.csv_file=Global.SintelSave
    else:
        import csv
        r_ik2=r_ik2.astype(int)
        try:
            #os.makedirs('superpixel-benchmark/output/EXP/' + str(Global.K_C_temp) + '/')
            os.makedirs(Global.Folder + str(Global.K_C_temp) + '/')
        except FileExistsError:
            with open(Global.Folder+str(Global.K_C_temp)+'/'+Global.csv_file+'.csv', "w") as f:
                writer = csv.writer(f)
                writer.writerows(r_ik2)
    Global.K_C=count
    if(Global.Plot):
        framePointsNew = framePointsNew.astype('uint8')
        plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
        plt.axis('off')
        fig.savefig('Frame_'+str(Global.csv_file)+'_' + str(Global.K_C) + 'SP.png', format='png', dpi=1200,bbox_inches='tight',transparent = True, pad_inches = 0)
        fig = plt.figure()
        framePointsNew2 = framePointsNew2.astype('uint8')
        plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
        fig.savefig('Frame_'+str(Global.csv_file)+'_' + str(Global.K_C) + 'SP_B.png', format='png', dpi=1200)
        framePointsNew3 = framePointsNew3.astype('uint8')
        plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
        fig.savefig('Frame_' + str(Global.csv_file) + '_' + str(Global.K_C) + 'SP_O.png', format='png', dpi=1200)
        # fig.savefig('Frame_' + str(Global.C_prior) + '_' + str(Global.K_C) + 'SP.png', format='png', dpi=1200)
        # fig = plt.figure()
        # framePointsNew2 = framePointsNew2.astype('uint8')
        # plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
        # fig.savefig('Frame_' + str(Global.C_prior) + '_' + str(Global.K_C) + 'SP_B.png', format='png', dpi=1200)
        # framePointsNew3 = framePointsNew3.astype('uint8')
        # plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
        # fig.savefig('Frame_' + str(Global.C_prior) + '_' + str(Global.K_C) + 'SP_O.png', format='png', dpi=1200)


    # fig = plt.figure()
    # framePointsNew2 = framePointsNew2.astype('uint8')
    # plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
    # fig.savefig('frame1_' + str(Global.K_C) + 'SP_White.png', format='png', dpi=1200)
    # fig = plt.figure()
    # framePointsNew3 = framePointsNew3.astype('uint8')
    # plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
    # fig.savefig('frame1_' + str(Global.K_C) + 'SP_Original.png', format='png', dpi=1200)


    r_ik = argmax[:,1].cpu().numpy()
    Global.K_C=r_ik.max()+2
    SigmaXY_cpu = SigmaXY.cpu().numpy()[:, 0:4]
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    C1_cpu = C1.cpu().numpy()[:, 0:2]

    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    framePointsNew2 = np.zeros((Global.HEIGHT, Global.WIDTH, 3))

    mean_value = np.zeros((Global.K_C + 1, 3))
    mean_value2 = np.zeros((Global.K_C + 1, 3))

    # Trueflow = np.expand_dims(C1[:, 10:12], 1)
    # Trueflow = Trueflow.reshape(-1, 1, 2)
    # flow = my_help.computeImg(Trueflow)
    # #
    # hsv = np.zeros((Global.K_C+1,1,3), dtype=np.uint8)
    # hsv[..., 1] = 255
    #
    # mag, ang = cv2.cartToPolar(C2[..., 0], C2[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)
        # mean_value2[i, 0] = flow[i, :, 0]
        # mean_value2[i, 1] = flow[i, :, 1]
        # mean_value2[i, 2] = flow[i, :, 2]
        #
    # np.save(str(Global.IMAGE1) + '_mean_RGB', mean_value)
    # np.save(str(Global.IMAGE1) + '_mean_XY', C1_cpu)
    # np.save(str(Global.IMAGE1) + '_SigmaXY', SigmaXY_cpu)
    # np.save(str(Global.IMAGE1) + '_Clusters', r_ik2)

    # framePointsNew3 = Global.frame0
    #
    # for i in range(0, Global.HEIGHT):
    #     for j in range(0, Global.WIDTH):
    #         if (i == 0):
    #             if (j == 0):
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #             elif (j == Global.WIDTH - 1):
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #             else:
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (
    #                         r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #
    #
    #         elif (i == Global.HEIGHT - 1):
    #             if (j == 0):
    #                 if ((r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #             elif (j == Global.WIDTH - 1):
    #                 if ((r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #
    #             else:
    #                 if ((r_ik2[i, j] == r_ik2[i - 1, j]) and (r_ik2[i, j] == r_ik2[i, j - 1]) and (
    #                         r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #
    #
    #         else:
    #             if (j == 0):
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i - 1, j]) and (
    #                         r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #
    #             elif (j == Global.WIDTH - 1):
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i - 1, j]) and (
    #                         r_ik2[i, j] == r_ik2[i, j - 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    #
    #
    #             else:
    #                 if ((r_ik2[i, j] == r_ik2[i + 1, j]) and (r_ik2[i, j] == r_ik2[i - 1, j]) and (
    #                         r_ik2[i, j] == r_ik2[i, j - 1]) and (r_ik2[i, j] == r_ik2[i, j + 1])):
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #
    #                 else:
    #                     ind = int(r_ik2[i, j])
    #                     framePointsNew[i, j, :] = mean_value[ind]
    #                     framePointsNew2[i, j, :] = mean_value2[ind]
    #                     # framePointsNew[i, j, :] = 0
    #                     # framePointsNew2[i, j, :] = 0
    #                     # framePointsNew3[i, j, :] = 0
    #                     a=0
    #
    # fig = plt.figure()
    # framePointsNew = framePointsNew.astype('uint8')
    # plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    # fig.savefig('frame1_' + str(file_name) + '_0_SP_Avg.png', format='png', dpi=1200)
    # # fig = plt.figure()
    # # framePointsNew2 = framePointsNew2.astype('uint8')
    # # plt.imshow((cv2.cvtColor(framePointsNew2, cv2.COLOR_BGR2RGB)))
    # # fig.savefig('frame1_' + str(Global.K_C) + 'SP_White.png', format='png', dpi=1200)
    # # fig = plt.figure()
    # # framePointsNew3 = framePointsNew3.astype('uint8')
    # # plt.imshow((cv2.cvtColor(framePointsNew3, cv2.COLOR_BGR2RGB)))
    # # fig.savefig('frame1_' + str(Global.K_C) + 'SP_Original.png', format='png', dpi=1200)


def KmeansSP3TF(X, H, Y):
    """Soft K-means algorithem over the location,intesity and the optical flow

    **Parameters**:
     - :math:`X[N,5]` - Data matrix  [Point number,(X,Y,L,A,B].
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].
     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    global SIGMA
    global SIGMA1
    global SIGMA2
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0., 0.],
                      [0., 0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., 0., Global.opt_scale]])

    SIGMA1 = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0.],
                       [0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., Global.int_scale]])

    SIGMA2 = np.array([[Global.loc_scale, 0., 0., 0., 0., 0., 0., 0.],
                       [0., Global.loc_scale, 0., 0., 0., 0., 0., 0.],
                       [0., 0., Global.int_scale, 0., 0., 0., 0., 0.],
                       [0., 0., 0., Global.int_scale, 0., 0., 0., 0.],
                       [0., 0., 0., 0., Global.int_scale, 0., 0., 0.],
                       [0., 0., 0., 0., 0., Global.int_scale, 0., 0.],
                     T  [0., 0., 0., 0., 0., 0., Global.int_scale, 0.],
                       [0., 0., 0., 0., 0., 0., 0., Global.int_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 853
    it = 0
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = Z / (Zmax)
    # Z[Z==0]=np.nan
    #    Z=np.prod(Z,axis=1)
    threshold = np.percentile(Z,
                              5)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0

    H_t = Variable(torch.from_numpy(H), requires_grad=False).cuda().float()
    Y_t = Variable(torch.from_numpy(Y).unsqueeze(2), requires_grad=True).cuda().float()
    r_ik_t = Variable(torch.from_numpy(r_ik), requires_grad=True).cuda().float()
    pi_t = Variable(torch.from_numpy(pi), requires_grad=True).cuda().float()
    X_t = Variable(torch.from_numpy(X), requires_grad=True).cuda().float()
    SIGMA = Variable(torch.from_numpy(SIGMA), requires_grad=True).cuda().float()
    SIGMA1 = Variable(torch.from_numpy(SIGMA1), requires_grad=True).cuda().float()
    logdet = Variable(torch.slogdet(SIGMA)[1], requires_grad=True).cuda()

    T1_G = torch.bmm(H_t.transpose(1, 2), Y_t)
    HH_G = torch.bmm(H_t.transpose(1, 2), H_t)
    YY_G = torch.bmm(Y_t.transpose(1, 2), Y_t)
    with torch.no_grad():
        H_t1 = H_t + 1
        T1_G = torch.bmm(H_t.transpose(1, 2), Y_t)
    print(T1_G)
    H_t.data = torch.zeros(H.shape)
    T1 = torch.ones(Global.N, 2, 1).cuda()
    torch.autograd.backward([T1_G], [T1])
    print(T1_G)

    index2_buffer = torch.zeros(Global.N).cuda()
    r_ikNew_buffer = torch.zeros((Global.N, Global.K_C + 1)).cuda().reshape(-1)
    r_ik_t_buffer = torch.zeros((r_ik_t.shape)).cuda().reshape(-1)
    alpha_prime = Global.ALPHA_T + (Global.D_T * Global.N) / (2.0)
    eta_prime = torch.zeros(Global.K_C + 1, 2).cuda()
    distances1_buffer = torch.zeros((Global.N, Global.K_C + 1)).cuda()
    distances2_buffer = torch.zeros((Global.N, Global.K_C + 1)).cuda()
    X_C_buffer = torch.zeros((Global.K_C + 1, Global.N, 5)).cuda()
    Y_H_buffer = torch.zeros((Global.K_C + 1, Global.N, 2)).cuda()

    while (it < maxIt):
        it += 1
        if (it % 2 == 0):
            print(it)
        C1 = EstimateSPTF(X_t, r_ik_t, pi_t)  # M-Step
        C2, _, _ = TFEstimateParams(H_t, Y_t, r_ik_t, norm_Weight, T1, HH, YY, alpha_prime, eta_prime)
        C = torch.cat([C1, C2], dim=1)
        Nk = torch.sum(r_ik_t, dim=0)
        N_ = X.shape[1]
        pi_t = Nk / N_
        prev_r_ik = r_ik_t.clone()
        # prev_r_ik[:, 0] = 0
        prev_r_ik_max = torch.argmax(prev_r_ik, dim=1)
        prev_r_ik_max = prev_r_ik_max.view((Global.HEIGHT, Global.WIDTH))
        r_ik_t = FindClosestClusterSP2TF(X_t, C1, C2, pi_t, r_ik_t, H_t, Y_t, Global.opt_scale, norm_Weight, logdet,
                                         distances1_buffer, distances2_buffer, X_C_buffer, Y_H_buffer)  # E-Step
        r_ik_t = my_connectivity.Change_pixel(prev_r_ik_max, r_ik_t, index2_buffer, r_ik_t_buffer, r_ikNew_buffer,
                                              it % 4)
        print("next")
        memory_usage()
    #    r_ik=np.argmax(r_ik,axis=1)
    #    r_ik2=np.reshape(r_ik,(HEIGHT,WIDTH))
    #    framePointsNew=np.zeros((HEIGHT,WIDTH,3))
    #    for i in range(0,HEIGHT):
    #        for j in range(0,WIDTH):
    #            if(r_ik2[i,j]==-1):
    #                framePointsNew[i,j,:]=0,0,0
    #            if(r_ik2[i,j]==-2):
    #                framePointsNew[i,j,:]=255,255,255
    #            if(r_ik2[i,j]>=0):
    #                ind=int(r_ik2[i,j])
    #                framePointsNew[i,j,:]=my_help.hex2rgb(colors[ind])
    #
    #    framePointsNew=framePointsNew.astype('uint8')
    #    plt.imshow(framePointsNew)
    r_ik_t[:, 0] = 0
    r_ik = r_ik_t.cpu().numpy()
    r_ik = np.argmax(r_ik, axis=1)
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]
    fig = plt.figure()
    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    fig.savefig('plot.svg', format='svg', dpi=1200)
    return C, r_ik


def KmeansSP(X):
    """Soft K-means algorithem over :math:`X,Y` and  :math:`I`
    **Parameters**:
     - frame0
    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    global SIGMA
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., Global.int_scale]])

    r_ik, pi = InitKmeansSP()
    maxIt = 50
    it = 0

    while (it < maxIt):
        it += 1

        C = EstimateSP(X, r_ik, pi)  # M-Step

        _, r_ik = FindClosestClusterSP(X, C, pi, r_ik)  # E-Step
        Nk = r_ik.sum(0)
        N_ = r_ik.shape[0]
        pi = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
        pi[0] = Global.PI_0

    r_ik = np.argmax(r_ik, axis=1)
    r_ik2 = np.reshape(r_ik, (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]

    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    return C, r_ik


def KmeansSPTF(X):
    """Soft K-means algorithem over :math:`X,Y` and  :math:`I`
    **Parameters**:
     - frame0
    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    global SIGMA
    SIGMA = np.array([[Global.loc_scale, 0., 0., 0., 0.],
                      [0., Global.loc_scale, 0., 0., 0.],
                      [0., 0., Global.int_scale, 0., 0.],
                      [0., 0., 0., Global.int_scale, 0.],
                      [0., 0., 0., 0., Global.int_scale]])

    r_ik, pi = InitKmeansSP()
    SIGMA = torch.from_numpy(SIGMA).cuda().float()
    r_ik_t = torch.from_numpy(r_ik).cuda().float()
    pi_t = torch.from_numpy(pi).cuda().float()
    X_t = torch.from_numpy(X).cuda().float()
    logdet = torch.slogdet(SIGMA)[1]
    maxIt = 20
    it = 0

    idx = np.arange(0, Global.N)
    idx = idx.reshape((Global.HEIGHT, Global.WIDTH))
    idx = my_help.blockshaped(idx, 2, 2)
    idx = idx.reshape(-1, 4)
    while (it < maxIt):
        if (it % 20 == 0):
            print(it)
        it += 1

        C = EstimateSPTF(X_t, r_ik_t, pi_t)  # M-Step
        prev_r_ik = deepcopy(r_ik_t)
        prev_r_ik[:, 0] = 0
        prev_r_ik = torch.argmax(prev_r_ik, dim=1)
        prev_r_ik = prev_r_ik.reshape((Global.HEIGHT, Global.WIDTH))
        _, r_ik_t = FindClosestClusterSPTF(X_t, C, pi_t, logdet)  # E-Step
        # r_ik_t = my_connectivity.Change_pixel(prev_r_ik, idx[:, it % 4], r_ik_t)

        Nk = torch.sum(r_ik_t, dim=0)
        N_ = r_ik_t.shape[0]
        pi_t = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
        pi_t[0] = Global.PI_0
    r_ik_t[:, 0] = 0
    r_ik_t = (torch.argmax(r_ik_t, dim=1)).cpu().numpy()
    r_ik2 = np.reshape(r_ik_t, (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))
    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]

    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    return C, r_ik_t


def Kmeans(H, Y):
    """Soft K-means algorithem over the projection created by :math:`H` over :math:`Y`
    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].
       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`
     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].
       :math:`Y[:,0]=-I_t`
    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = np.ones(Z.shape)

    threshold = np.percentile(Z,
                              15)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0
    nan_idxH = np.where(~H.all(axis=2))[0]
    r_ik, pi = InitKmeans(H, Y)

    # H = torch.from_numpy(H).type(device)
    # Y = torch.from_numpy(Y).type(device)
    # r_ik = torch.from_numpy(r_ik).type(device)
    # pi = torch.from_numpy(pi).type(device)
    if (Global.HARD_EM):
        idx = np.argmax(r_ik, axis=1)
        c = r_ik.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        idx = (u < c).argmax(axis=1)
        r_ik2 = np.zeros(r_ik.shape)
        r_ik2[np.arange(r_ik2.shape[0]), idx] = 1
        r_ik = r_ik2
        a = np.zeros(Global.K_C + 1)
        np.put(a, 0, 1.0)
        r_ik[nan_idxH] = a
    maxIt = 700
    it = 0

    while (it < maxIt):
        it += 1
        mu_max, sigmaMax, Cmax = EstimateParams(H, Y, r_ik, norm_Weight)  # M-Step
        C = mu_max
        # C[0] = [100, 100]
        prev_r_ik = deepcopy(r_ik)
        prev_r_ik = np.argmax(prev_r_ik, axis=1)
        prev_r_ik = prev_r_ik.reshape((Global.HEIGHT, Global.WIDTH))
        _, r_ik = FindClosestCluster(H, Y, C, r_ik, pi, norm_Weight)  # E-Step
        if (Global.HARD_EM):
            for i in range(0, Global.N):
                a = np.zeros(Global.K_C + 1)
                value = my_connectivity.Change_pixel(prev_r_ik, i, r_ik)
                np.put(a, int(value), 1.0)
                r_ik[i] = a
            # idx = np.argmax(r_ik, axis=1)
            # c = r_ik.cumsum(axis=1)
            # u = np.random.rand(len(c), 1)
            # idx = (u < c).argmax(axis=1)
            # r_ik2 = np.zeros(r_ik.shape)
            # r_ik2[np.arange(r_ik2.shape[0]), idx] = 1
            # r_ik = r_ik2
            # a = np.zeros(Global.K_C + 1)
            # np.put(a, 0, 1.0)
            # r_ik[nan_idxH] = a
        Nk = r_ik.sum(0)
        N_ = Y.shape[0]
        pi = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
        pi[0] = Global.PI_0

    # New AreaEstimate
    #    h_c=np.tensordot(C,H,axes=(1,2))
    #    Y_H=Y-h_c
    #    Y_H/=(0.9999999*norm_Weight+0.0000001)
    #    distances=(-0.5)*np.einsum("ijk,ijk->ij",Y_H,Y_H)
    #    _,logdet = np.linalg.slogdet(M_0)
    #    distances+=np.log(pi[:,np.newaxis])
    #    threshold=np.percentile(distances,5,axis=1)
    #    distances[distances<threshold[:,np.newaxis]]=np.nan
    # New Area

    r_ik = np.argmax(r_ik, axis=1)
    # r_ik[nan_idxH]=0
    r_ik2 = np.reshape(r_ik[:Global.N], (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))

    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            elif (r_ik2[i, j] == 0):
                framePointsNew[i, j, :] = 255, 255, 255
            elif (r_ik2[i, j] >= 1):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = my_help.hex2rgb(Global.colors[ind])

    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow(framePointsNew)
    # plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    return C, r_ik, Cmax


def KmeansTF(H, Y):
    """Soft K-means algorithem over the projection created by :math:`H` over :math:`Y`
    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].
       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`
     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].
       :math:`Y[:,0]=-I_t`
    **Returns**:
     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]
     - Clusters[N,K+1]- Clusters [Number of points,Number of clusters + outlayer] .
    """
    Z = np.linalg.norm(H, axis=2)
    Zmax = np.max(Z, axis=0)
    norm_Weight = np.ones(Z.shape)

    threshold = np.percentile(Z,
                              15)  # Maybe threshold for zeros only? need to check.. asuumption that larage magnitude equal to good mesaurment?
    Z[Z < threshold] = np.nan
    H = H / Z[:, :, np.newaxis]
    Y = Y / Z
    Y[np.isnan(Y)] = 0
    H[np.isnan(H)] = 0
    Y[np.isinf(Y)] = 0
    H[np.isinf(H)] = 0
    nan_idxH = np.where(~H.all(axis=2))[0]
    r_ik, pi = InitKmeans(H, Y)

    # H = torch.from_numpy(H).type(device)
    # Y = torch.from_numpy(Y).type(device)
    # r_ik = torch.from_numpy(r_ik).type(device)
    # pi = torch.from_numpy(pi).type(device)
    # if (Global.HARD_EM):
    #
    #     idx = np.argmax(r_ik, axis=1)
    #     c = r_ik.cumsum(axis=1)
    #     u = np.random.rand(len(c), 1)
    #     idx = (u < c).argmax(axis=1)
    #     r_ik2 = np.zeros(r_ik.shape)
    #     r_ik2[np.arange(r_ik2.shape[0]), idx] = 1
    #     r_ik = r_ik2
    #     a = np.zeros(Global.K_C + 1)
    #     np.put(a, 0, 1.0)
    #     r_ik[nan_idxH] = a
    maxIt = 20
    it = 0

    # H = torch.from_numpy(H).type(device).share_memory_()
    # Y = torch.from_numpy(Y).type(device).share_memory_()
    # r_ik = torch.from_numpy(r_ik).type(device).share_memory_()
    # pi = torch.from_numpy(pi).type(device).share_memory_()

    H_t = torch.tensor(H).cuda().float()
    Y_t = torch.from_numpy(Y).cuda().unsqueeze(2).float()
    r_ik_t = torch.from_numpy(r_ik).cuda().float()
    pi_t = torch.from_numpy(pi).cuda().float()
    idx = np.arange(0, Global.N)
    idx = idx.reshape((Global.HEIGHT, Global.WIDTH))
    idx = my_help.blockshaped(idx, 2, 2)
    idx = idx.reshape(-1, 4)

    T1 = torch.bmm(H_t.transpose(1, 2), Y_t)
    HH = torch.bmm(H_t.transpose(1, 2), H_t)
    YY = torch.bmm(Y_t.transpose(1, 2), Y_t)
    alpha_prime = Global.ALPHA_T + (Global.D_T * Global.N) / (2.0)
    num_processes = 5
    processes = []
    # manager = mp.Manager()
    # return_dict = manager.dict()
    while (it < maxIt):
        if (it % 20 == 0):
            print(it)
        it += 1
        jobs = []
        # for id in range(num_processes):
        #     p = mp.Process(target=TFEstimateParams2, args=(H,Y,r_ik[:,id],id,processes))
        #     p.start()
        #     jobs.append(p)
        # for p in processes:
        #     p.join()

        mu_max, sigmaMax, Cmax = TFEstimateParams(H_t, Y_t, r_ik_t, norm_Weight, T1, HH, YY, alpha_prime)  # M-Step
        C = mu_max
        prev_r_ik = deepcopy(r_ik_t)
        prev_r_ik[:, 0] = 0
        prev_r_ik = torch.argmax(prev_r_ik, dim=1)
        prev_r_ik = prev_r_ik.reshape((Global.HEIGHT, Global.WIDTH))
        r_ik_t = FindClosestClusterTF(H_t, Y_t, C, r_ik_t, pi_t, norm_Weight)  # E-Step

        # if(Global.HARD_EM):

        # ind=my_connectivity.Change_pixel(prev_r_ik,idx[:,it%4],r_ik_t)

        # r_ik_t=my_connectivity.Change_pixel(prev_r_ik,idx[:,it%4],r_ik_t)

        # valsNew=torch.argmax(r_ik_t,dim=1)
        # r_ik_t=torch.zeros((r_ik_t.shape)).cuda()
        # r_ik_t[np.arange(0, Global.N), valsNew.view(-1)]=1 # new value of every pixel
        # prev_rik_val=torch.zeros((r_ik_t.shape)).cuda()
        # prev_rik_val[np.arange(0, Global.N), prev_r_ik.view(-1)]=1
        # r_ik_t[ind==0]=prev_rik_val[ind==0] #Keep the previous values of r_ik for those pixels we cant chagne
        # r_ik_t[valsNew.view(-1) == 0] = prev_rik_val[valsNew.view(-1) == 0] #Keep the previous layer if argmax is outlayer
        # r_ik_t[valsNew.view(-1)==0,0]=1 #Outlier should be declared even if we cant change the pixel label

        # idx = np.argmax(r_ik, axis=1)
        # c = r_ik.cumsum(axis=1)
        # u = np.random.rand(len(c), 1)
        # idx = (u < c).argmax(axis=1)
        # r_ik2 = np.zeros(r_ik.shape)
        # r_ik2[np.arange(r_ik2.shape[0]), idx] = 1
        # r_ik = r_ik2
        # a = np.zeros(Global.K_C + 1)
        # np.put(a, 0, 1.0)
        # r_ik[nan_idxH] = a
        Nk = r_ik_t.sum(0)
        N_ = Y_t.shape[0]
        pi_t = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
        pi[0] = Global.PI_0

    # New AreaEstimate
    #    h_c=np.tensordot(C,H,axes=(1,2))
    #    Y_H=Y-h_c
    #    Y_H/=(0.9999999*norm_Weight+0.0000001)
    #    distances=(-0.5)*np.einsum("ijk,ijk->ij",Y_H,Y_H)
    #    _,logdet = np.linalg.slogdet(M_0)
    #    distances+=np.log(pi[:,np.newaxis])
    #    threshold=np.percentile(distances,5,axis=1)
    #    distances[distances<threshold[:,np.newaxis]]=np.nan
    # New Area
    r_ik_t[:, 0] = 0
    r_ik = r_ik_t.cpu().numpy()
    r_ik = np.argmax(r_ik, axis=1)
    # r_ik[nan_idxH]=0
    r_ik2 = np.reshape(r_ik[:Global.N], (Global.HEIGHT, Global.WIDTH))
    framePointsNew = np.zeros((Global.HEIGHT, Global.WIDTH, 3))

    mean_value = np.zeros((Global.K_C + 1, 3))
    for i in range(0, Global.K_C + 1):
        mean_value[i] = np.mean(Global.frame0[r_ik2 == i], axis=0)  ##need args where
    for i in range(0, Global.HEIGHT):
        for j in range(0, Global.WIDTH):
            if (r_ik2[i, j] == -1):
                framePointsNew[i, j, :] = 0, 0, 0
            if (r_ik2[i, j] == -2):
                framePointsNew[i, j, :] = 255, 255, 255
            if (r_ik2[i, j] >= 0):
                ind = int(r_ik2[i, j])
                framePointsNew[i, j, :] = mean_value[ind]

    framePointsNew = framePointsNew.astype('uint8')
    plt.imshow(framePointsNew)
    # plt.imshow((cv2.cvtColor(framePointsNew, cv2.COLOR_BGR2RGB)))
    return C, r_ik, Cmax


""" 

Find Closest Cluster Function

"""


def FindClosestCluster(H, Y, C, r_ik, pi, norm_Weight):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`Help_Functions.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold
u
    """

    h_c = np.tensordot(C, H, axes=(1, 2))  # TODO   :Check
    Y_H = Y - h_c
    distances = (-0.5) * np.einsum("ijk,ijk->ij", Y_H, Y_H)
    _, logdet = np.linalg.slogdet(Global.M_0)
    distances[1:Global.K_C + 1] *= 1
    distances[1:Global.K_C + 1] += np.log(1)

    distances[0] *= 0.000001  # param for outliers    /
    distances[0] += np.log(0.000001)  # variance for outliers

    distances += np.log(pi[:, np.newaxis])
    distances[np.isnan(distances)] = -np.inf
    r_ik = (my_help.softmax(distances, 0)).T

    return distances, r_ik


def FindClosestClusterSP2(X, C1, C2, pi, r_ik, H, Y, opt_scale, norm_Weight):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters
     [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """

    h_c = np.tensordot(C2, H, axes=(1, 2))  # TODO   :Check
    Y_H = Y - h_c
    Y_H_SIGMA = Y_H * (1.0 / opt_scale)
    # Y_H[Y_H==0.5]/=0.00001
    # distances=(-0.5)*(y_2[np.newaxis]-2*y_h_c+h_c_2)[:,:,0]
    distances2 = (-0.5) * np.einsum("ijk,ijk->ij", Y_H_SIGMA, Y_H)

    X_C = X[np.newaxis] - C1[:, np.newaxis]
    X_C_SIGMA = X_C * (1.0 / np.diagonal(SIGMA1))
    distances1 = (-0.5) * np.einsum("ijk,ijk->ij", X_C_SIGMA, X_C)

    # distances2[0]/=1000

    # distances1[0]=(-0.5)*np.sum(X_C_SIGMA0*X_C,axis=2).T[0]

    threshold = np.percentile(norm_Weight.sum(axis=1), 1)
    distances = distances1 + distances2
    distances[:, norm_Weight.sum(axis=1) < threshold] = distances1[:, norm_Weight.sum(axis=1) < threshold]
    _, logdet = np.linalg.slogdet(SIGMA)

    distances -= logdet  # variance

    distances += np.log(pi[:, np.newaxis])
    distances[np.isnan(distances)] = -np.inf

    (my_help.softmax(distances, 0))

    return distances, r_ik


def FindClosestClusterSP2TF(X, Y, opt_scale,int_scale,logdet, H_5,c1_temp,c2_temp,pi_temp,SigmaXY,X_C_SIGMA,distances1,distances2,sum,maskall):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters
     [Number of pסoints,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """

    Y = Y.transpose(0, 1)
    distances1.zero_()
    distances2.zero_()
    # HC2=torch.mul(c2_temp.reshape(-1, 1, 5, 2), H_5.reshape(Global.N, -1, 5, 2))
    # HC2=HC2.reshape(-1,5,2)
    # HC2[:,:,0].add_=(HC2[:,:,1])
    # HC2=HC2[:,:,0]
    # Y_H = torch.add(Y, -HC2).reshape(-1,3,5)

    Y_H=torch.add(Global.TrueFlow.unsqueeze(1),-c2_temp.reshape(-1,5,2))
    Y_H_SIGMA = torch.mul(Y_H, (1.0 / opt_scale))

    X_C = torch.add(X.unsqueeze(1), -c1_temp.reshape(-1, 5, 5))
    X_C_SIGMA[:, :, 0] = torch.add(torch.mul(X_C[:, :, 0], SigmaXY[:, :, 0]),torch.mul(X_C[:, :, 0], SigmaXY[:, :, 2]))
    X_C_SIGMA[:, :, 1] = torch.add(torch.mul(X_C[:, :, 0], SigmaXY[:, :, 1]),torch.mul(X_C[:, :, 1], SigmaXY[:, :, 3]))
    X_C_SIGMA[:,:,2:5] = torch.mul(X_C[:,:,2:5],(1.0/int_scale))
    mulX=torch.neg(torch.mul(X_C, X_C_SIGMA)).transpose(1,2)
    distances1.add_(mulX[:,0]).add_(mulX[:,1]).add_(mulX[:,2]).add_(mulX[:,3]).add_(mulX[:,4])
    mulY= torch.neg(torch.mul(Y_H, Y_H_SIGMA))
    distances2.add_(mulY[:,:, 0]).add_(mulY[:, :,1])
    distances2[maskall]=distances1[maskall]
    distances2.add_(distances1)
    distances2.add_(torch.neg(logdet.view(-1, 5)))
    distances2.add_(torch.log(pi_temp.view(-1, 5)))
    (my_help.softmaxTF(distances2, 1,sum))

        # Y=Y.transpose(0,1)
        #
        # HC2=torch.mul(H_5,c2_temp.reshape(-1,5,2))
        # HC2[:,:,0].add_=(HC2[:,:,1])
        # HC2=HC2[:,:,0].transpose(0,1)
        # Y_H = torch.add(Y, -HC2).view(-1, 3).transpose(0, 1)
        # X_C=torch.add(X.unsqueeze(1), -c1_temp.view(-1, 5, 5)).transpose(0,1).reshape(-1,5).transpose(0,1)
        # X_C_SIGMA = torch.mul(X_C, (1.0 / torch.diagonal(SIGMA1)).unsqueeze(1))
        # X_C=X_C.transpose(0,1).reshape(5,-1,5).transpose(0,1)
        # X_C_SIGMA=X_C_SIGMA.transpose(0,1).reshape(5,-1,5).transpose(0,1)
        # X_C_SIGMA.view(-1,5,5)[:,:,0]=torch.add(torch.mul(X_C[:,:,0],SigmaXY[:,:,0]),torch.mul(X_C[:,:,0],SigmaXY[:,:,2]))
        # X_C_SIGMA.view(-1,5,5)[:,:,1]=torch.add(torch.mul(X_C[:,:,0],SigmaXY[:,:,1]),torch.mul(X_C[:,:,1],SigmaXY[:,:,3]))
        # X_C=X_C.transpose(0,1).reshape(-1,5).transpose(0,1)
        # X_C_SIGMA=X_C_SIGMA.transpose(0,1).reshape(-1,5).transpose(0,1)
        # Y_H_SIGMA = torch.mul(Y_H, (1.0 / opt_scale))
        # distances1 = torch.neg(torch.sum(torch.mul(X_C, X_C_SIGMA), dim=0).view(5, -1))  # factor of 2
        # distances2 = torch.neg(torch.sum(torch.mul(Y_H, Y_H_SIGMA), dim=0)).view(5, -1)  # factor of 2
        # distances2.add_(distances1)
        # distances2.add_(torch.neg(logdet.view(-1,5).transpose(0,1)))
        # distances2.add_(torch.log(pi_temp.view(-1,5).transpose(0,1)))
        # r_ik = (my_help.softmaxTF(distances2, 0)).transpose(1, 2)[0]


def FindClosestClusterSP_2Frames(X,logdet,c1_temp,pi_temp,SigmaXY,X_C_SIGMA,sum,c_idx,c_idx_9,c_idx_25,distances2,r_ik_5,neig,sumP,X_C,X_C_SIGMA_buf):

    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters
     [Number of pסoints,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """
    torch.add(X.unsqueeze(1), torch.neg(c1_temp.reshape(-1, Global.neig_num, Global.D_)),out=X_C)
    #X_C_SIGMA[:,:,0:2]=torch.add(torch.mul(X_C[:,:,0].unsqueeze(2),SigmaXY[:,:,0:2]),torch.mul(X_C[:,:,1].unsqueeze(2),SigmaXY[:,:,2:4]))
    torch.mul(X_C[:, :, 0].unsqueeze(2), SigmaXY[:, :, 0:2],out=X_C_SIGMA_buf)
    torch.addcmul(X_C_SIGMA_buf,1,X_C[:,:,1].unsqueeze(2),SigmaXY[:,:,2:4],out=X_C_SIGMA[:,:,0:2])
    X_C_SIGMA[:, :, 2:5] = torch.mul(X_C[:, :, 2:5], Global.SIGMA_INT)

    if(Global.D_12):
        X_C_SIGMA[:,:,5:7]=torch.add(torch.mul(X_C[:,:,5].unsqueeze(2),SigmaXY[:,:,4:6]),torch.mul(X_C[:,:,6].unsqueeze(2),SigmaXY[:,:,6:8]))
        X_C_SIGMA[:,:,7:12]=torch.mul(X_C[:,:,7:12],Global.SIGMA_INT_FLOW)

    # torch.bmm(-X_C.reshape(-1, 1, 5),X_C_SIGMA.reshape(-1,5,1 ),out=r_ik_5)
    # r_ik_5=r_ik_5.view(-1,Global.neig_num)
    torch.mul(-X_C.view(-1, Global.neig_num,Global.D_),X_C_SIGMA.view(-1,Global.neig_num,Global.D_),out=distances2)
    distances2=distances2.view(-1,Global.neig_num,Global.D_)
    torch.sum(distances2,2,out=r_ik_5)

    r_ik_5.add_(torch.neg(logdet.reshape(-1, Global.neig_num)))
    r_ik_5.add_(torch.log(pi_temp.reshape(-1, Global.neig_num)))



    c_neig = c_idx_25.reshape(-1, Global.potts_area).float()

    torch.add(c_neig.unsqueeze(1), -c_idx.reshape(-1, Global.neig_num).unsqueeze(2).float(),out=neig)
    torch.sum((neig!=0).float(),2,out=sumP)
    r_ik_5.add_(-(Global.Beta_P*sumP))


    (my_help.softmaxTF(r_ik_5, 1,sum))
    return distances2



def FindClosestClusterTF(H, Y, C, r_ik, pi, norm_Weight):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`Help_Functions.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """

    # h_c=rrch.einsum("ijk,mk->mij", (H, C))

    Y_H = Y[:, :, 0] - (
        torch.bmm(H.unsqueeze(0).repeat(Global.K_C + 1, 1, 1, 1).view(Global.K_C + 1, -1, 2), C.unsqueeze(2)).view(
            Global.K_C + 1, Global.N, -1))
    # Y_H = Y[:, :, 0]-torch.mm(H.view(-1, 2), C.transpose(0, 1)).view(Global.K_C + 1, -1, 3)
    distances = -torch.bmm(Y_H.view(Global.N * (Global.K_C + 1), -1, 1).transpose(1, 2),
                           Y_H.view(Global.N * (Global.K_C + 1), -1, 1)).view(Global.K_C + 1, -1)
    # _, logdet = torch.slogdet(Global.M_0_T)
    # distances[1:Global.K_C + 1]*=Global.one_T
    # distances[1:Global.K_C + 1]+=Global.log1_T
    distances[0] = torch.mul(distances[0], Global.small_T)  # param for outliers    /
    distances[0] = torch.add(distances[0], Global.logSmall_T)
    distances = torch.add(distances, torch.log(pi.unsqueeze(1)))
    # if(distances[torch.isnan(distances)].shape[0]):
    #     distances[torch.isnan(distances)]=torch.add(distances[torch.isnan(distances)], -Global.inf_T)
    r_ik = (my_help.softmaxTF(distances, 0)).transpose(1, 2)

    # torch.mul(distances[0],Global.small_T.transpoes(0,1))  # param for outliers    /
    # torch.add(distances[0],Global.logSmall_T.transpose(0,1))   # variance for outliers
    # torch.add(distances,torch.log(pi.unsqueeze(1)))

    # distances[0] *= Global.small_T  # param for outliers    /
    # distances[0] += Global.logSmall_T  # variance for outliers
    # distances += torch.log(pi.unsqueeze(1))
    return r_ik[0]


def FindClosestClusterSP2(X, C1, C2, pi, r_ik, H, Y, opt_scale, norm_Weight):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters
     [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """

    h_c = np.tensordot(C2, H, axes=(1, 2))  # TODO   :Check
    Y_H = Y - h_c
    Y_H_SIGMA = Y_H * (1.0 / opt_scale)
    # Y_H[Y_H==0.5]/=0.00001
    # distances=(-0.5)*(y_2[np.newaxis]-2*y_h_c+h_c_2)[:,:,0]
    distances2 = (-0.5) * np.einsum("ijk,ijk->ij", Y_H_SIGMA, Y_H)

    X_C = X[np.newaxis] - C1[:, np.newaxis]
    X_C_SIGMA = X_C * (1.0 / np.diagonal(SIGMA1))
    distances1 = (-0.5) * np.einsum("ijk,ijk->ij", X_C_SIGMA, X_C)

    # distances2[0]/=1000

    # distances1[0]=(-0.5)*np.sum(X_C_SIGMA0*X_C,axis=2).T[0]

    threshold = np.percentile(norm_Weight.sum(axis=1), 1)
    distances = distances1 + distances2
    distances[:, norm_Weight.sum(axis=1) < threshold] = distances1[:, norm_Weight.sum(axis=1) < threshold]
    _, logdet = np.linalg.slogdet(SIGMA)

    distances -= logdet  # variance

    distances += np.log(pi[:, np.newaxis])
    distances[np.isnan(distances)] = -np.inf

    r_ik = (my_help.softmax(distances, 0)).T

    return distances, r_ik


def FindClosestClusterSP(X, C, pi, r_ik):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """

    X_C = X[:, np.newaxis] - C[np.newaxis]
    X_C = X[np.newaxis] - C[:, np.newaxis]
    X_C_SIGMA = X_C * (1.0 / np.diagonal(SIGMA))
    distances = (-0.5) * np.einsum("ijk,ijk->ij", X_C_SIGMA, X_C)
    # distances[0]=(-0.5)*np.sum(X_C_SIGMA0*X_C,axis=2).T[0]
    _, logdet = np.linalg.slogdet(SIGMA)
    # _,logdet0 = np.linalg.slogdet(np.eye(4)*1000)
    # distances[1:K_C+1]-=logdet # variance
    distances -= logdet  # variance
    # distances[0]-=logdet0
    distances += np.log(pi[:, np.newaxis])
    distances[np.isnan(distances)] = -np.inf

    r_ik = (my_help.softmax(distances, 0)).T
    return distances, r_ik


def FindClosestClusterSPTF(X, C, pi, logdet):
    """Computes the distances of the projections points for eace centroid and normalize it using :meth:`AngleImpl.softmax`,

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - :math:`C[K+1,D]` - Centroids [Number of clusters + outlayer,Dimenstion of data]

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer]

    **Returns**:
     - Distances[K,N*]- Distances [Number of clusters + outlayer,Number of points after threshold] .

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: Changes the values only for points that passed through the threshold

    """
    X_C = X.unsqueeze(0) - C.unsqueeze(1)
    X_C_SIGMA = torch.mul(X_C, (1.0 / torch.diagonal(SIGMA)))
    distances = (-0.5) * torch.bmm(X_C_SIGMA.view(-1, 1, 5), X_C.view(-1, 5, 1)).view(Global.K_C + 1, Global.N, -1)
    # distances[0]=(-0.5)*np.sum(X_C_SIGMA0*X_C,axis=2).T[0]
    # _,logdet0 = np.linalg.slogdet(np.eye(4)*1000)
    # distances[1:K_C+1]-=logdet # variance
    distances = torch.add(distances, -logdet)[:, :, 0]  # variance
    # distances[0]-=logdet0
    distances = torch.add(distances, torch.log(pi.unsqueeze(1)))
    # distances[torch.isnan(distances)] = -Global.inf_T #TODO: change!
    # distances[:,0]=-Global.inf_T
    r_ik = (my_help.softmaxTF(distances, 0)).transpose(1, 2)

    return distances, r_ik[0]

0
def InitKmeans(H, Y):
    """Initialize the clutsters probability for each point.

    **Parameters**:
     -

    **Returns**:
     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: The probobility randomize unirformly

    .. note:: Changed the initialzation similar for adjacent pixels (3x3)

    """

    """ No outlier 
    r_ik=np.ones((H.shape[0],K_C))

    for i in range(0,H.shape[0]):
        r_ik[i]=np.random.dirichlet(np.ones(K_C),size=1)

    Nk=r_ik.sum(0)
    N_=Y.size
    pi=Nk/N_

    No oultier """

    r_ik = np.zeros((H.shape[0], Global.K_C + 1))

    # TODO: Add this lines for random init
    for i in range(0, H.shape[0]):
        r_ik[i, 1:Global.K_C + 1] = (np.random.dirichlet(np.ones(Global.K_C), size=1)) / (1.0 - Global.PI_0)
    r_ik[:, 0] = Global.PI_0

    # kH = int(Global.HEIGHT / 9)
    # kW = int(Global.WIDTH / 9)
    # indx=np.arange(Global.N).reshape((Global.HEIGHT,Global.WIDTH))
    # indx=my_help.blockshaped(indx,kH,kW)
    #
    # for i in range(0,Global.K_C):
    #     a = np.zeros(Global.K_C+1)
    #     np.put(a, i+1, 1.0)
    #     r_ik[indx[i].reshape(kH*kW)]=a
    #

    Nk = r_ik.sum(0)
    N_ = Y.shape[0]
    pi = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
    pi[0] = Global.PI_0

    return r_ik, pi


def InitKmeansSP():
    """Initialize the clutsters probability for each point.

    **Parameters**:
     -

    **Returns**:
     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .

    .. note:: The probobility randomize unirformly

    .. note:: Changed the initialzation similar for adjacent pixels (3x3)

    """

    #    r_ik=np.ones((N,K_C+1))
    #    r_ik=np.reshape(r_ik,((HEIGHT,WIDTH,K_C+1)))
    #    for i in range(0,HEIGHT-3,3):
    #        for j in range(0,WIDTH-3,3):
    #            r_ik[i:i+3,j:j+3]=np.random.dirichlet(np.ones(K_C+1),size=1)
    #    r_ik=np.reshape(r_ik,(N,K_C+1)
    x, y = np.mgrid[:Global.HEIGHT, :Global.WIDTH]
    X = np.array((x.ravel(), y.ravel()))
    s=np.int(np.sqrt(Global.N/Global.K_C))
    a=np.int(Global.WIDTH/s)
    b=np.int(Global.HEIGHT/s)
    a_2=a+2


    h, w = np.mgrid[Global.HEIGHT/(2*(b+1)):Global.HEIGHT-1:(Global.HEIGHT/(b+1)),Global.WIDTH/(2*(a+1)):Global.WIDTH-1:(Global.WIDTH/(a+1))]
    h2, w2 = np.mgrid[Global.HEIGHT/(2*(b+1)):Global.HEIGHT-1:(Global.HEIGHT/(b+1)),Global.WIDTH/(2*(a_2+1)):Global.WIDTH-1:(Global.WIDTH/(a_2+1))]

    C = np.array((h.ravel(), w.ravel()),dtype=np.float).T
    C2 = np.array((h2.ravel(), w2.ravel()),dtype=np.float).T





    width=(C[2][1]-C[1][1])*0.5


    # print(C.shape[0])
    # for i in range(C.shape[0]):
    #     if(i==35):
    #         a=5
    #     C_temp = C[i]
    #     if((i//(w.shape[1]))%2==1):
    #         if(i%w.shape[1]==(w.shape[1]-1)):
    #             C_temp[1] = C_temp[1] + width/2
    #             C = np.append(C, np.array([C_temp]), axis=0)
    #         C[i][1]=C[i][1]-width

    print(C.shape[0])

    C_0 = np.array([[Global.HEIGHT * 5, Global.HEIGHT * 5]])
    C = np.append(C_0, C, axis=0)
    voronoi_kdtree = cKDTree(C)
    extraPoints = X.transpose()
    test_point_dist, test_point_regions = voronoi_kdtree.query(extraPoints)


    Global.K_C = C.shape[0]
    Global.K_C_ORIGINAL = C.shape[0]
    Global.RegSize=Global.N/Global.K_C

    Global.A_prior=Global.N/(Global.K_C)
    #r_ik = (my_help.softmax(distances[:Global.K_C + 1], 0)).T
    r_ik=test_point_regions

    #    Nk=r_ik.sum(0)
    #    N_=X.shape[1]
    #    pi=Nk/N_

    # Nk = r_ik.sum(0)
    # N_ = r_ik.shape[0]
    # pi = Nk * (1 - Global.PI_0) / (N_ - Nk[0])
    # pi[0] = Global.PI_0

    return r_ik, r_ik


"""

Estimation Fucntion

"""


def TFEstimateParams(H, Y, argmax, T1_temp, HH, YY, alpha_prime, eta_prime,T1,T2,T3):
    #r_ik=r_ik.transpose(0,1)

    T1.zero_()
    T2.zero_()
    T3.zero_()
    T1.index_add_(0,argmax,T1_temp)
    T2.index_add_(0,argmax,HH)
    T3.index_add_(0,argmax,YY[:,0])
    T2=T2.reshape(-1,2,2)
    T3=T3.unsqueeze(1)

    # T1_1 = torch.mm(r_ik, T1_temp)
    # T2_1 = torch.mm(r_ik, HH).view(-1, 2, 2)
    # T3_1 = torch.mm(r_ik, YY)

    m_prime = torch.add(T2, Global.M_T.unsqueeze(0))
    det = torch.reciprocal (0.00000001+torch.add(torch.mul(m_prime[:, 0, 0], m_prime[:, 1, 1]), -torch.mul(m_prime[:, 0, 1], m_prime[:, 1, 0])))
    m_prime_i = torch.tensor(m_prime)

    ad = m_prime.reshape(-1, 4)
    m_prime_i[:, 0, 0] = torch.mul(ad[:, 3], det)
    m_prime_i[:, 0, 1] = torch.mul(-ad[:, 1], det)
    m_prime_i[:, 1, 0] = torch.mul(-ad[:, 2], det)
    m_prime_i[:, 1, 1] = torch.mul(ad[:, 0], det)

    M_ETA_T1 = torch.add(Global.M_ETA_T.unsqueeze(0), T1.unsqueeze(2))


    eta_prime00 = torch.mul(m_prime_i[:, 0, 0], M_ETA_T1[:, 0, 0])
    eta_prime10 = torch.mul(m_prime_i[:, 1, 0], M_ETA_T1[:, 0, 0])
    eta_prime01 = torch.mul(m_prime_i[:, 0, 1], M_ETA_T1[:, 1, 0])
    eta_prime11 = torch.mul(m_prime_i[:, 1, 1], M_ETA_T1[:, 1, 0])
    eta_prime[:, 0] = torch.add(eta_prime00, eta_prime01)
    eta_prime[:, 1] = torch.add(eta_prime10, eta_prime11)

    temp1= torch.mul(eta_prime[:,0].pow(2),m_prime[:,0,0])
    temp2= torch.mul(torch.add(m_prime[:,0,1],m_prime[:,1,0]),torch.mul(eta_prime[:,0],eta_prime[:,1]))
    temp3 =torch.mul(eta_prime[:,1].pow(2),m_prime[:,1,1])
    temp4= torch.add(torch.add(temp1,temp2),temp3)
    beta_prime = torch.add(Global.BETA_T , (0.5) * (torch.add(torch.add(T3 ,Global.ETA_M_ETA_T) , temp4)))

    mu_max, sigma_max, c_max = my_sample.SampleNGTF(alpha_prime, beta_prime, eta_prime)
    Nk=torch.zeros(Global.K_C+1).cuda()
    Nk.index_add_(0, argmax, Global.ones)
    X2=torch.zeros(Global.K_C+1,2).cuda()
    Nk = Nk + 0.0000000001
    X2.index_add_(0,argmax,Global.TrueFlow)
    mu_max = torch.div(X2, Nk.unsqueeze(1))


    return mu_max, sigma_max, c_max


def TFEstimateParams2(H, Y, r_ik, id, return_dict):
    # Global.initVariables()
    # r_ik=r_ik.unsqueeze(0)
    # HT = H.transpose(1,2)0
    # T1 = torch.einsum("ijk,ik->ij", [HT, Y])
    # T1 = torch.einsum("ij,ik", [T1, r_ik])
    # T1 = T1.transpose(0, 1)
    # HH = torch.einsum("ijk,ikm->ijm", [HT, H])
    # T2 = torch.einsum("ijk,im", [HH, r_ik])
    # del HH
    # T2 = T2.transpose(0, 2).transpose(1, 2)
    # YY = torch.einsum("ijk,ij->ik", [Y.unsqueeze(2), Y])
    # T3 = torch.einsum("ij,im", [YY, r_ik])
    # del YY
    # T3 = T3.transpose(0, 1)
    # # TODO:CHANGE
    # N_used = torch.einsum("ij->", [r_ik]).float()
    # # TODO:CHANGE
    #
    #
    # m_prime = (Global.M_T.unsqueeze(0) + T2)
    # del T2
    # m_prime_i=torch.inverse(m_prime[0]).unsqueeze(0)
    #
    #
    # eta_prime = torch.einsum("ijk,ikm->ijm",[m_prime_i,(Global.M_ETA_T.unsqueeze(0)+ T1.unsqueeze(2))])
    # del T1
    # eta_primeT = eta_prime.transpose(1,2)
    #
    #
    # temp1 = torch.einsum("ijk,imj->imj", [m_prime, eta_primeT])
    # beta_prime = Global.BETA_T + (0.5) * (T3 + Global.ETA_M_ETA_T - torch.einsum("ijk,ikj->ij", [temp1, eta_prime]))
    # del T3
    #
    # mu_max = eta_prime[:, :, 0]
    # alpha_prime = Global.ALPHA_T + (Global.D_T * N_used) / (2.0)
    # c_max = (alpha_prime[0] - 0.5) / (beta_prime[0])
    # eye_D = torch.eye(Global.D).float().cuda(async=True)
    # sigma_max = torch.inverse(c_max * eye_D)
    #
    # return 1
    return_dict[id] = [mu_max, sigma_max, c_max]


def EstimateParams(H, Y, r_ik, norm_Weight):
    """Estimate the parmeters :math:`\\mu , \\Sigma , C` using Angle derivations.

    **Parameters**:
     - :math:`H[N,d,D]` - Projection matrix  [Point number,Dimension of projection, Dimenstion of data].

       :math:`H[:,:,0]=I_x` :math:`H[:,:,1]=I_y`

     - :math:`Y[N,d]` - Projection point [Point number,Dimension of projection].

       :math:`Y[:,0]=-I_t`

     - Clusters[N,K]- Clusters [Number of points,Number of clusters + outlayer] .


    **Returns**:
     - :math:`\\mu [K+1,D]` - Mean [Number of clusters + outlayer,Dimenstion of data] .

     - :math:`\\Sigma [K+1,D,D]` - Covariance  [Number of clusters + outlayer,Dimenstion of data,Dimenstion od data] .

     - :math:`C [K+1,D]` - C [Number of clusters + outlayer,Dimenstion of data] .

    .. note:: Estimation made by using weights defined by the clusters probability


    """

    #    H=H*(norm_Weight[:,:,np.newaxis])
    #    Y=Y*(norm_Weight)

    HT = np.swapaxes(H, axis1=1, axis2=2)
    weights = r_ik
    # weights=np.ones(r_ik.shape)

    """ Start Part 1 """

    T1 = np.einsum("ijk,ik->ij", HT, Y)
    T1 = np.einsum("ij,ik", T1, weights)
    HH = np.einsum("ijk,ikm->ijm", HT, H)
    T2 = np.einsum("ijk,im", HH, weights)
    YY = np.einsum("ijk,ij->ik", Y[:, :, np.newaxis], Y)
    T3 = np.einsum("ij,im", YY, weights)

    # end = time.time()
    # print(end - start)
    T1 = T1.T
    T2 = T2.T
    T3 = T3.T

    """ End Part 1 """
    # T2=np.sum(weights[:,:,np.newaxis,np.newaxis]*(HT@H)[np.newaxis],axis=1)
    # T3=np.sum(weights[:,:,np.newaxis]*Y**2,axis=1)

    # HH_T=np.einsum("ijk,ikm->ijm",H,HT)
    # HH_T=np.linalg.pinv(HH_T)
    # HH_T=np.eye(3)
    # HH_T=np.repeat(HH_T[:, :, np.newaxis], 307200, axis=2)
    # Z = np.linalg.norm(H, axis=2)
    # HH_T=HH_T.T * Z[:, :, np.newaxis]
    # #HH_T=HH_T.T
    #
    #
    # H_HH_T=np.einsum("ijk,ikm->ijm",HT,HH_T)
    # T1=np.einsum("ijk,ik->ij",H_HH_T,Y)
    # T1=np.einsum("ij,ik",T1,weights)
    # T1=T1.T
    #
    # T2=np.einsum("ijk,ikm->ijm",H_HH_T,H)
    # T2 = np.einsum("ijk,im", T2, weights)
    # T2=T2.T
    #
    # T3 = np.einsum("ijk,ijm->ikm", Y[:, :, np.newaxis],HH_T)
    # T3= np.einsum("ijk,ik->ij",T3,Y)
    # T3 = np.einsum("ij,im", T3, weights)
    # T3=T3.T

    # N_used = Y.shape[0]
    # TODO:CHANGE
    N_used = np.sum(weights)
    N_used /= 4.0
    # TODO:CHANGE

    alpha_prime = Global.ALPHA + float((Global.D * N_used) / (2))
    m_prime = Global.M[np.newaxis] + T2
    eta_prime = inv(m_prime) @ ((Global.M @ Global.ETA)[np.newaxis] + T1[:, :, np.newaxis])
    eta_primeT = np.swapaxes(eta_prime, axis1=1, axis2=2)

    temp1 = np.einsum("ijk,imj->imj", m_prime, eta_primeT)
    beta_prime = Global.BETA + (0.5) * (
                T3 + ((Global.ETA.T) @ Global.M @ (Global.ETA)) - np.einsum("ijk,ikj->ij", temp1, eta_prime))

    mu_max, sigma_max, c_max = my_sample.SampleNG(alpha_prime, beta_prime, eta_prime)

    return mu_max, sigma_max, c_max


def EstimateSP(X, r_ik, pi):
    Nk = r_ik.sum(0)
    C = (np.tensordot(X, r_ik, (0, 0)).T) / Nk[:, np.newaxis]

    return C


def EstimateSPTF(X,argmax, Sigma,SigmaInv,Nk,X1,X2_00,X2_01,X2_11):
    Nk.zero_()
    X1.zero_()
    X2_00.zero_()
    X2_01.zero_()
    X2_11.zero_()

    Nk.index_add_(0, argmax, Global.ones)
    Nk = Nk + 0.0000000001
    X1.index_add_(0,argmax,X)
    C = torch.div(X1, Nk.unsqueeze(1))

    #Nk =torch.sum(r_ik, dim=0) + 0.0000000001
    #X1 = (torch.mm(r_ik.transpose(0, 1), X))


    mul=torch.pow(X[:,0],2)
    X2_00.index_add_(0,argmax,mul)

    mul=torch.mul(X[:,0],X[:,1])
    X2_01.index_add_(0,argmax,mul)

    mul=torch.pow(X[:,1],2)
    X2_11.index_add_(0,argmax,mul)



    # X2_00=torch.mm(r_ik.transpose(0,1),torch.pow(X[:,0],2).unsqueeze(1))[:,0]
    # X2_01=torch.mm(r_ik.transpose(0,1),torch.mul(X[:,0],X[:,1]).unsqueeze(1))[:,0]
    # X2_11=torch.mm(r_ik.transpose(0,1),torch.pow(X[:,1],2).unsqueeze(1))[:,0]

    Sigma00=torch.add(X2_00,-torch.div(torch.pow(X1[:,0],2),Nk))
    Sigma01=torch.add(X2_01,-torch.div(torch.mul(X1[:,0],X1[:,1]),Nk))
    Sigma11=torch.add(X2_11,-torch.div(torch.pow(X1[:,1],2),Nk))
    Sigma[:,0]=torch.div(torch.add(Sigma00,Global.PSI_prior[0]),torch.add(Nk,Global.NI_prior))
    Sigma[:,1]=torch.div((Sigma01),torch.add(Nk,Global.NI_prior))
    Sigma[:,2]=Sigma[:,1]
    Sigma[:,3]=torch.div(torch.add(Sigma11,Global.PSI_prior[3]),torch.add(Nk,Global.NI_prior))


    # Sigma[:,0]=Global.loc_scale
    # Sigma[:,1]=0
    # Sigma[:, 2] =0
    # Sigma[:, 3] =Global.loc_scale


    det=torch.reciprocal(torch.add(torch.mul(Sigma[:,0],Sigma[:,3]),-torch.mul(Sigma[:,1],Sigma[:,2])))
    det[(det<=0).nonzero()]=0.00001
    SigmaInv[:, 0] = torch.mul(Sigma[:, 3], det)
    SigmaInv[:, 1] = torch.mul(-Sigma[:, 1], det)
    SigmaInv[:, 2] = torch.mul(-Sigma[:, 2], det)
    SigmaInv[:, 3] = torch.mul(Sigma[:, 0], det)
    SIGMAxylab[:,0:2,0:2]=SigmaInv.view(-1,2,2)




    return C,Sigma,SigmaInv,Nk

def EstimateSP_2Frames(X,loc,argmax, Sigma, SigmaInv, Nk, X1, X2_00, X2_01, X2_11,init,Nk_s,X1_s,X2_00_s,X2_01_s,X2_11_s,SigmaXY_s,SigmaInv_s,it,max_it): #Nk_r,X1_r, X2_00_r, X2_01_r,X2_11_r,SigmaXY_r,SigmaXY_l,SigmaInv_r,SigmaInv_l):
    Nk.zero_()
    Nk_s.zero_()
    X1.zero_()
    # X1_s.zero_()
    X2_00.zero_()
    X2_01.zero_()
    X2_11.zero_()
    # X2_00_s.zero_()
    # X2_01_s.zero_()
    # X2_11_s.zero_()
    if(init==True):
        argmax1=argmax[:,0].clone()
    else:
        argmax1=argmax[:,1].long()
    argmax=argmax[:,0]
    if(Global.HARD_EM or init==True):
        Nk.index_add_(0, argmax, Global.ones)
        Nk = Nk + 0.0000000001
        # Nk_s.index_add_(0, argmax1, Global.ones)
        # Nk_s = Nk_s + 0.0000000001

        X1.index_add_(0,argmax,X)
        # X1_s.index_add_(0,argmax1,X)

        C = torch.div(X1, Nk.unsqueeze(1))
        # C_s = torch.div(X1_s, Nk_s.unsqueeze(1))


    mul=torch.pow(loc[:,0],2)
    X2_00.index_add_(0,argmax,mul)
    # X2_00_s.index_add_(0,argmax1,mul)


    mul=torch.mul(loc[:,0],loc[:,1])
    X2_01.index_add_(0,argmax,mul)
    # X2_01_s.index_add_(0,argmax1,mul)


    mul=torch.pow(loc[:,1],2)
    X2_11.index_add_(0,argmax,mul)
    # X2_11_s.index_add_(0,argmax1,mul)


    Sigma00=torch.add(X2_00,-torch.div(torch.pow(X1[:,0],2),Nk))
    Sigma01=torch.add(X2_01,-torch.div(torch.mul(X1[:,0],X1[:,1]),Nk))
    Sigma11=torch.add(X2_11,-torch.div(torch.pow(X1[:,1],2),Nk))

    # Sigma00_s = torch.add(X2_00_s, -torch.div(torch.pow(X1_s[:, 0], 2), Nk_s))
    # Sigma01_s = torch.add(X2_01_s, -torch.div(torch.mul(X1_s[:, 0], X1_s[:, 1]), Nk_s))
    # Sigma11_s = torch.add(X2_11_s, -torch.div(torch.pow(X1_s[:, 1], 2), Nk_s))
    #

    Global.C_prior = 999+ (torch.mul(0,-Global.split_lvl[0:Nk.shape[0]]))
    a_prior=torch.mul(torch.pow(2,Global.split_lvl[0:Nk.shape[0]]),Global.A_prior)
    # if(max_it-it<25):
    #     a_prior = Nk
    #     Global.C_prior = 8 + (torch.mul(0, -Global.split_lvl[0:Nk.shape[0]]))
    if(max_it-it<50 or (it%25<8)):
        Global.C_prior = 49+ (torch.mul(0,-Global.split_lvl[0:Nk.shape[0]]))
        a_prior=Nk
        # Global.C_prior[Nk>(Global.RegSize*1.2)]= 1000
        # Global.C_prior[Nk > (Global.RegSize * 1.5)] = 2000
        # Global.C_prior[Nk>(Global.RegSize*1.8)]= 3000
        # Global.C_prior[Nk>(Global.RegSize*2.0)]= 4000
        # Global.C_prior[Nk > (Global.RegSize * 2.5)] = 6000
        # # Global.C_prior[Nk<(Global.RegSize*0.95)]= 55
        # # Global.C_prior[Nk<(Global.RegSize*0.75)]= 50
        # Global.C_prior[Nk<(Global.RegSize*0.5)]= 5
        # Global.C_prior[Nk<(Global.RegSize*0.25)]=2



    Global.psi_prior=torch.mul(torch.pow(a_prior,2).unsqueeze(1),torch.eye(2).reshape(-1,4).cuda())
    Global.ni_prior=(Global.C_prior*a_prior)-3






    # Sigma[:,0]=torch.div(torch.add(Sigma00,Global.PSI_prior[0]),torch.add(Nk,Global.NI_prior))
    # Sigma[:,1]=torch.div((Sigma01),torch.add(Nk,Global.NI_prior))
    # Sigma[:,2]=Sigma[:,1]
    # Sigma[:,3]=torch.div(torch.add(Sigma11,Global.PSI_prior[3]),torch.add(Nk,Global.NI_prior))

    Sigma[:, 0] = torch.div(torch.add(Sigma00, Global.psi_prior[:,0]), torch.add(Nk, Global.ni_prior))
    Sigma[:, 1] = torch.div((Sigma01), torch.add(Nk, Global.ni_prior))
    Sigma[:, 2] = Sigma[:, 1]
    Sigma[:, 3] = torch.div(torch.add(Sigma11, Global.psi_prior[:,3]), torch.add(Nk, Global.ni_prior))

    # SigmaXY_s[:, 0] = torch.div(torch.add(Sigma00_s, Global.psi_prior_sons[:,0]), torch.add(Nk_s, (Global.ni_prior_sons)))
    # SigmaXY_s[:, 1] = torch.div((Sigma01_s), torch.add(Nk_s,((Global.ni_prior_sons))))
    # SigmaXY_s[:, 2] = SigmaXY_s[:, 1]
    # SigmaXY_s[:, 3] = torch.div(torch.add(Sigma11_s, Global.psi_prior_sons[:,3]), torch.add(Nk_s, ((Global.ni_prior_sons))))


    det=torch.reciprocal(torch.add(torch.mul(Sigma[:,0],Sigma[:,3]),-torch.mul(Sigma[:,1],Sigma[:,2])))
    # det_s = torch.reciprocal(torch.add(torch.mul(SigmaXY_s[:, 0], SigmaXY_s[:, 3]), -torch.mul(SigmaXY_s[:, 1], SigmaXY_s[:, 2])))


    det[(det <= 0).nonzero()] = 0.00001
    # det_s[(det <= 0).nonzero()] = 0.00001

    SigmaInv[:, 0] = torch.mul(Sigma[:, 3], det)
    SigmaInv[:, 1] = torch.mul(-Sigma[:, 1], det)
    SigmaInv[:, 2] = torch.mul(-Sigma[:, 2], det)
    SigmaInv[:, 3] = torch.mul(Sigma[:, 0], det)

    # SigmaInv_s[:, 0] = torch.mul(SigmaXY_s[:, 3], det_s)
    # SigmaInv_s[:, 1] = torch.mul(-SigmaXY_s[:, 1], det_s)
    # SigmaInv_s[:, 2] = torch.mul(-SigmaXY_s[:, 2], det_s)
    # SigmaInv_s[:, 3] = torch.mul(SigmaXY_s[:, 0], det_s)


    SIGMAxylab[:,0:2,0:2]=Sigma[:,0:4].view(-1,2,2)
    # SIGMAxylab_s[:,0:2,0:2]=SigmaInv_s[:,0:4].view(-1,2,2)




    # X2_00_s.zero_()G
    # X2_01_s.zero_()
    # X2_11_s.zero_()


    if(Global.D_12):
        X2_00.zero_()
        X2_01.zero_()
        X2_11.zero_()
        mul = torch.pow(loc[:, 2], 2)
        X2_00.index_add_(0, argmax, mul)
        X2_00_s.index_add_(0,argmax1,mul)


        mul = torch.mul(loc[:, 2], loc[:, 3])
        X2_01.index_add_(0, argmax, mul)
        X2_01_s.index_add_(0,argmax1,mul)



        mul = torch.pow(loc[:, 3], 2)
        X2_11.index_add_(0, argmax, mul)
        X2_11_s.index_add_(0,argmax1,mul)



        Sigma00 = torch.add(X2_00, -torch.div(torch.pow(X1[:, 5], 2), Nk))
        Sigma01 = torch.add(X2_01, -torch.div(torch.mul(X1[:, 5], X1[:, 6]), Nk))
        Sigma11 = torch.add(X2_11, -torch.div(torch.pow(X1[:, 6], 2), Nk))

        Sigma00_s = torch.add(X2_00_s, -torch.div(torch.pow(X1_s[:, 5], 2), Nk_s))
        Sigma01_s = torch.add(X2_01_s, -torch.div(torch.mul(X1_s[:, 5], X1_s[:, 6]), Nk_s))
        Sigma11_s = torch.add(X2_11_s, -torch.div(torch.pow(X1_s[:, 6], 2), Nk_s))



        Sigma[:, 4] = torch.div(torch.add(Sigma00, Global.PSI_prior[0]), torch.add(Nk, Global.NI_prior))
        Sigma[:, 5] = torch.div((Sigma01), torch.add(Nk, Global.NI_prior))
        Sigma[:, 6] = Sigma[:, 5]
        Sigma[:, 7] = torch.div(torch.add(Sigma11, Global.PSI_prior[3]), torch.add(Nk, Global.NI_prior))

        SigmaXY_s[:, 4] = torch.div(torch.add(Sigma00_s, Global.PSI_prior[0] / 4),torch.add(Nk_s, ((Global.NI_prior + 3) / 2) - 3))
        SigmaXY_s[:, 5] = torch.div((Sigma01_s), torch.add(Nk_s, ((Global.NI_prior + 3) / 2) - 3))
        SigmaXY_s[:, 6] = SigmaXY_s[:, 5]
        SigmaXY_s[:, 7] = torch.div(torch.add(Sigma11_s, Global.PSI_prior[3] / 4),torch.add(Nk_s, ((Global.NI_prior + 3) / 2) - 3))


        det = torch.reciprocal(torch.add(torch.mul(Sigma[:, 4], Sigma[:, 7]), -torch.mul(Sigma[:, 5], Sigma[:, 6])))
        det_s = torch.reciprocal(torch.add(torch.mul(SigmaXY_s[:, 4], SigmaXY_s[:, 7]), -torch.mul(SigmaXY_s[:, 5], SigmaXY_s[:, 6])))

        det[(det <= 0).nonzero()] = 0.00001
        det_s[(det <= 0).nonzero()] = 0.00001

        SigmaInv[:, 4] = torch.mul(Sigma[:, 7], det)
        SigmaInv[:, 5] = torch.mul(-Sigma[:, 5], det)
        SigmaInv[:, 6] = torch.mul(-Sigma[:, 6], det)
        SigmaInv[:, 7] = torch.mul(Sigma[:, 4], det)

        SigmaInv_s[:, 4] = torch.mul(SigmaXY_s[:, 7], det_s)
        SigmaInv_s[:, 5] = torch.mul(-SigmaXY_s[:, 5], det_s)
        SigmaInv_s[:, 6] = torch.mul(-SigmaXY_s[:, 6], det_s)
        SigmaInv_s[:, 7] = torch.mul(SigmaXY_s[:, 4], det_s)



        SIGMAxylab[:, 5:7, 5:7] = SigmaInv[:,4:8].view(-1, 2, 2)
        SIGMAxylab_s[:, 5:7, 5:7] = SigmaInv_s[:,4:8].view(-1, 2, 2)


    #return C,Sigma,SigmaInv,Nk,[C_s,SigmaXY_s,SigmaInv_s,Nk_s,X1_s]
    logdet=torch.log(torch.mul(torch.reciprocal(det),Global.detInt))
    return C,logdet,Sigma,SigmaInv,Nk,[Nk_s,SigmaXY_s,SigmaInv_s,Nk_s,X1_s]




"""
************************************************************************************************************************************************************

                                                                        Gradient Funtions

************************************************************************************************************************************************************
"""


def Compute_Gradient(frame0, frame1):
    """Computing the gradient between 2 frames

    **Parameters**:
     - frame0 - The first frame

     - frame1 - The second frame

    **Returns**:
     - :math:`[frame0_x,frame0_y,frame0_t]` - Gradient between the frames

    .. note:: Using gaussian blur to the frame

    .. todo:: Finding the optimeal bluring in terms of speed

    """

    #    frame0 =0.85*frame0+ cv2.Laplacian(frame0,5)
    #    frame1 =0.85*frame1+ cv2.Laplacian(frame1,5)

    frame0 = frame0.astype(np.float32)
    frame1 = frame1.astype(np.float32)

    frame0 = cv2.GaussianBlur(frame0, (9, 9), 7)
    frame1 = cv2.GaussianBlur(frame1, (9, 9), 7)

    kernel = np.ones((2, 2), np.float32) / 4

    frame0 = cv2.filter2D(frame0, cv2.CV_32F, kernel)
    frame1 = cv2.filter2D(frame1, cv2.CV_32F, kernel)

    frame0_x = cv2.filter2D(frame0, cv2.CV_32F, Global.dx)
    frame0_y = cv2.filter2D(frame0, cv2.CV_32F, Global.dy)

    frame0_t = frame1 - frame0
    return [frame0_x, frame0_y, frame0_t]


def Compute_Gradient2(frame0, frame1):
    """Computing the gradient between 2 frames

    **Parameters**:
     - frame0 - The first frame

     - frame1 - The second frame

    **Returns**:
     - :math:`[frame0_x,frame0_y,frame0_t]` - Gradient between the frames

    .. note:: Using gaussian blur to the frame

    .. todo:: Finding the optimeal bluring in terms of speed

    """

    #    frame0 =0.85*frame0+ cv2.Laplacian(frame0,5)
    #    frame1 =0.85*frame1+ cv2.Laplacian(frame1,5)

    frame0 = frame0.astype(np.float32)
    frame1 = frame1.astype(np.float32)
    frame0_a = cv2.GaussianBlur(frame0, (9, 9), 7)
    frame1_a = cv2.GaussianBlur(frame1, (9, 9), 7)

    frame0 = np.swapaxes(frame0, 0, 2)
    frame0 = np.swapaxes(frame0, 1, 2)
    frame1 = np.swapaxes(frame1, 0, 2)
    frame1 = np.swapaxes(frame1, 1, 2)
    frame0_t = torch.from_numpy(frame0)
    frame1_t = torch.from_numpy(frame1)
    Gblur_x = cv2.getGaussianKernel(sigma=9, ksize=7)
    Gblur_y = cv2.getGaussianKernel(sigma=9, ksize=7)
    Gblur_xy = Gblur_x * Gblur_y.T
    print(Gblur_xy)
    Gblur_xy = torch.from_numpy(Gblur_xy)

    x = torch.nn.Conv2d(3, 3, [7, 7], padding=(3, 3))  # in_channels = 10, out_channels = 10
    frame0_t = x(frame0_t.unsqueeze_(0))
    frame1_t = x(frame1_t.unsqueeze_(0))
    frame0_t = frame0_t.detach().numpy()
    frame1_t = frame1_t.detach().numpy()
    frame0_t = np.swapaxes(frame0_t, 1, 3)
    frame0_t = np.swapaxes(frame0_t, 1, 2)
    frame1_t = np.swapaxes(frame1_t, 1, 3)
    frame1_t = np.swapaxes(frame1_t, 1, 2)
    plt.imshow(cv2.cvtColor(frame0_t[0], cv2.COLOR_BGR2RGB))

    frame0_t = frame1 - frame0
    return [frame0_x, frame0_y, frame0_t]


"""
************************************************************************************************************************************************************

                                                                        Main Funtions

************************************************************************************************************************************************************
"""


def AngleImpl():
    """First try, sampling theta

    **Parameters**:
     -

    **Returns**:
     - Figure with the clusterd points

    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.

    """

    global N
    global Y
    global X
    global H

    plt.figure(1, "Roy")
    mu, sigma, _, _ = SampleNG()
    X = SampleX(mu, sigma)
    Theta = sampleTheta()
    H = Calc_H(Theta)
    Y = Calc_Y(X, H)
    ProjPoints = Calc_ProjPoints(Y, H)

    maxXY = np.amax([np.amax(X[0, :]), np.amax(X[1, :])])
    minXY = np.amin([np.amin(X[0, :]), np.amin(X[1, :])])

    # Set Axes
    fig, ax = plt.subplots(2, 2)
    for (m, n), subplot in np.ndenumerate(ax):
        subplot.set_xlim(minXY, maxXY)
        subplot.set_ylim(minXY, maxXY)
        plt.axis("equal")

    plt.clf()
    ax = plt.subplot(3, 2, 1)
    ax.set_title("Data")
    PlotContour(mu, sigma, X, 'k', 'Original')
    PlotPoints(X, 'r', 'X - Data')

    ax = plt.subplot(3, 2, 2)
    ax.set_title("UV Lines")

    PlotPoints(X, 'r', 'X - Data')
    PlotPoints(ProjPoints, 'k', 'ProjPoints - Projection')
    PlotLines(X, ProjPoints)
    ax.set_title("Likelihood Function")

    ax = plt.subplot(3, 2, 5, projection='3d')
    mu_sample, sigma_sample, mu_max, sigma_max, c_sample, c_max = EstimateParams(H, Y)
    mu_sample = np.reshape(mu_sample, D)

    ax = plt.subplot(3, 2, 3)
    ax.set_title("Sample From Postirior")
    PlotContour(mu, sigma, X, 'k', 'Original')
    PlotContour(mu_sample, sigma_sample, X, 'b', 'Sample')
    PlotPoints(X, 'r', 'X - Data')

    ax = plt.subplot(3, 2, 4)
    ax.set_title("Argmax of Postirior")
    PlotContour(mu, sigma, X, 'k', 'Original')
    PlotContour(mu_max, sigma_max, X, 'b', 'Argmax')
    PlotPoints(X, 'r', 'X - Data')

    print("MU ", mu)
    print("Sigma: ", sigma)
    print("MU Sample", mu_sample)
    print("Sigma Sample: ", sigma_sample)
    print("MU Max: ", mu_max)
    print("Sigma MAx: ", sigma_max)


def ArttificalFlow():
    """Second try, one motion

    **Parameters**:
     -

    **Returns**:
     - Figure with the clusterd points

    .. math:: \alpha \beta
    .. note:: Changes the following global parmeters:
       :math:`\\mu , \\Sigma , C`.

    """

    priorSigma = (Global.ALPHA - 0.5) / (Global.BETA)
    #    priorVarianceMu=inv(priorSigma*M)
    #    priorVarianceSigma=(ALPHA/(BETA**2))
    print(Back.BLUE)
    print("Real Mu:", MU, "Real Sigma:", SIGMA)
    print("Real C:", C)
    print(Back.GREEN)
    print("Prior Mu:", ETA, "Prior Sigma: ", priorSigma)
    # print("Prior variance over MU: " ,priorVarianceMu , " Prior variance over C : " ,priorVarianceSigma)
    print(Style.RESET_ALL)
    N = HEIGHT * WIDTH
    Y = np.zeros((d, N))
    H = np.zeros((N, d, D))
    X = np.zeros((D, N))
    X = SampleX(MU, SIGMA)
    maxXY = np.amax([np.amax(X[0, :]), np.amax(X[1, :])])
    minXY = np.amin([np.amin(X[0, :]), np.amin(X[1, :])])

    # Set Axes
    fig, ax = plt.subplots(3, 2)
    for (m, n), subplot in np.ndenumerate(ax):
        subplot.set_xlim(minXY, maxXY)
        subplot.set_ylim(minXY, maxXY)
        plt.axis("equal")

    # UV=np.reshape(X,(D,height,width))
    U = np.reshape(X[:, 0], (HEIGHT, WIDTH))
    V = np.reshape(X[:, 1], (HEIGHT, WIDTH))
    U = U.astype(np.float32)
    V = V.astype(np.float32)

    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    xx = xx - U
    yy = yy - V
    plt.clf()
    plt.subplot(321)
    plt.imshow(frame0, cmap='gray')
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    frame1 = cv2.remap(frame0, xx, yy, cv2.INTER_CUBIC)
    plt.subplot(322)
    plt.imshow(frame1, cmap='gray')
    [Ix, Iy, It] = Compute_Gradient(frame0, frame1)
    Ix = np.ravel(Ix)
    Iy = np.ravel(Iy)
    It = np.ravel(It)

    plt.subplot(323)
    PlotContour(MU, SIGMA, X, 'k', 'Original')
    PlotPoints(X, 'r', 'X - Data')

    H[:, 0, 0] = Ix
    H[:, 0, 1] = Iy
    Y[0, :] = -It
    EstimateParams(H, Y)
    plt.subplot(325)
    mu_sample, sigma_sample, mu_max, sigma_max, c_sample, c_max = EstimateParams(H, Y)
    mu_sample = np.reshape(mu_sample, D)
    plt.subplot(324)
    PlotContour(mu_max, sigma_max, X, 'b', 'Argmax')
    PlotContour(MU, SIGMA, X, 'k', 'Original')
    PlotPoints(X, 'r', 'X - Data')

    print(Back.RED)
    print("Mu Max: ", mu_max, "Sigma Max : ", sigma_max)
    print("C Max: ", c_max)
    print(Style.RESET_ALL)


def memory_usage():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and torch.is_tensor(obj.data):
            print(type(obj), obj.size())


def SuperPixels2Frames():
    X0 = my_help.Create_DataMatrix(Global.frame0)
    X1 = my_help.Create_DataMatrix(Global.frame1)
    X1[:,0:2]+=Global.TrueFlow
    X=np.append(X0,X1,axis=1)
    X=np.append(X,Global.TrueFlow,axis=1)
    loc= np.append(X0[:,0:2],X1[:,0:2], axis=1)
    Kmeans2Frames(X,loc)


def SuperPixelsSplitMerge():
    X0 = my_help.Create_DataMatrix(Global.frame0)
    if(Global.D_12):
        X1 = my_help.Create_DataMatrix(Global.frame1)
        X1[:,0:2]+=Global.TrueFlow.cpu().numpy()
        X=np.append(X0,X1,axis=1)
        X=np.append(X,Global.TrueFlow.cpu().numpy(),axis=1)
        loc = np.append(X0[:, 0:2], X1[:, 0:2], axis=1)
        KmeansSplitMerge(X,loc)
    else:
        KmeansSplitMerge(X0,X0[:,0:2])



def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector






"""-
************************************************************************************************************************************************************

                                                                            Main

************************************************************************************************************************************************************
"""

if __name__ == "__main__":
    np.random.seed(34)
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    plt.interactive(False)
    Global.Sintel=False
    import os
    count_debug=0
    for p in range(0,6):
        if(p==0):
            Global.K_C_temp=550
            Global.K_C_LOW=550
            Global.K_C_HIGH =650
        if(p==1):
            Global.K_C_temp=500
            Global.K_C_LOW= 550
            Global.K_C_HIGH =650
        if (p == 2):
            Global.K_C_temp = 950
            Global.K_C_HIGH = 1000
            Global.K_C_LOW= 900
        if (p == 3):
            Global.K_C_temp = 900
            Global.K_C_HIGH = 1000
            Global.K_C_LOW = 900
        if (p == 4):
            Global.K_C_temp = 350
            Global.K_C_HIGH = 350
            Global.K_C_LOW = 250
        if (p == 5):
            Global.K_C_temp = 250
            Global.K_C_HIGH = 350
            Global.K_C_LOW = 250
        for k in range(0,1):
            # if(k==0):
            #     directory = os.fsencode('benchmark_opt/train/')
            #     Global.Folder='benchmark_opt/train/'
            # if(k==1):
            #     directory = os.fsencode('benchmark_opt/test/')
            #     Global.Folder='benchmark_opt/test/'
            # if (k == 2):
            #     directory = os.fsencode('benchmark_opt/val/')
            #     Global.Folder='benchmark_opt/val/'
            directory = os.fsencode('benchmark_opt/test/')
            Global.Folder = 'benchmark_opt/test/'
            # directory = os.fsencode('PAPER-VIS/')
            # Global.Folder = 'PAPER-VIS/'
            #Global.K_C_temp = 350
            #Global.C_prior = 100
            if(Global.Sintel):
                rootdir='Sintel/training/final/'
                Global.Folder = 'benchmark_opt/test/'
                for subdir, dirs, files in os.walk(rootdir):
                    for file in files:
                        Global.SintelSave=subdir[len(rootdir):]+'_'+file
                        Global.IMAGE1 =subdir+'/'+file
                        if torch.cuda.is_available():
                            print("Init Global")
                            Global.initVariables()
                            print("Init Data")
                            Global.split_lvl.zero_()
                            SuperPixelsSplitMerge()


            else:
                for file in os.listdir(directory):
                    filename = os.fsdecode(file)
                    if (filename.endswith(".jpg") or filename.endswith(".png")):
                        count_debug +=1
                        if(count_debug==12):
                            aa=12
                            #1/0
                        Global.csv_file=filename[:len(filename)-4]
                        Global.IMAGE1=Global.Folder+filename
                        print(torch.cuda.is_available())
                        if torch.cuda.is_available():
                            print("Init Global")
                            Global.initVariables()
                            Global.K_C=Global.K_C_temp
                            print("Init Data")
                            # Global.C_prior = 100
                            Global.split_lvl.zero_()
                        # plt.figure()
                        # plt.subplot(221)
                        # plt.imshow(cv2.cvtColor(Global.frame0, cv2.COLOR_BGR2RGB))
                        # plt.subplot(222)
                        # print("Optical Flow")
                        ##PyramidFlow()
                        # plt.subplot(223)
                        # print("SuperPixel V1")
                        # ArttificalFlowK2()
                        # plt.figure()
                        # SuperPixelsV1TF()
                        # plt.figure()
                        #SuperPixelsV2TF()
                        #SuperPixelsCuda()
                        #SuperPixels2Frames()
                            SuperPixelsSplitMerge()
                        # SuperPixelsV3TF()
                        # plt.figure()
                        # plt.subplot(224)
                        # print("SuperPixel V2")
                        # SuperPixelsV2()
                        # SuperPixelsV2TF()
                        # plt.figure()
                        # print("PyramidFlow")
                        # ArttificalFlowK2()
                        # ArttificalFlowPyramid()
                            plt.show(block=True)

