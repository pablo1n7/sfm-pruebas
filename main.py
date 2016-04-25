#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 4: 3D Scene Reconstruction Using Structure From Motion
    An app to detect and extract structure from motion on a pair of images
    using stereo vision. We will assume that the two images have been taken
    with the same camera, of which we know the internal camera parameters. If
    these parameters are not known, use calibrate.py to estimate them.
    The result is a point cloud that shows the 3D real-world coordinates
    of points in the scene.
"""

import numpy as np
import reconstruccion3D as r3D

def main():
    K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4,1006.81/4, 0, 0, 1]]).reshape(3, 3)
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    ruta = "img/monumento/"
    a = r3D.Rescontruccion3D(ruta,K,d)


if __name__ == '__main__':
    main()