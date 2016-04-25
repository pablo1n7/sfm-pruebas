import cv2
import os 
import pcl
import numpy as np
from vista import *
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

class Rescontruccion3D(object):
    """docstring for rescontruccion3D"""
    vistas = []
    camaras = []
    pts3D= []
    K = None
    R = None
    t = None
    E = None
    F = None
    FMask = None
    d = None
    ruta = None
    
    #vistasProcesadas

    def __init__(self, ruta, K, d):
        self.K = K
        self.d = d
        self.ruta = ruta
        directorio = os.walk(self.ruta)
        for root, dirs, files in directorio:
            for fichero in files:
                (nombreFichero, extension) = os.path.splitext(fichero)
                if(extension == ".png" or extension == ".JPG"):
                    self.vistas.append(Vista(self.ruta+nombreFichero+extension))

        #if self.vistas.len<2:
         #   print "Error - no se pudieron localizar las vistas"
        #else:
        """la inicializacion se realiza con las dos primera imagenes, apartir de ahi se 
        deberia ir incrementando la nube 3D a partir de las demas vistas"""
        self.vistas[0].buscar_feature(self.vistas[1]) #la primera inicializacion
        self.obtener_matriz_fundamental(self.vistas[0].features[self.vistas[1]].match_puntos_uno,self.vistas[0].features[self.vistas[1]].match_puntos_dos)
        self.obtener_matriz_esencial()
        puntos_homogenios_uno,puntos_homogenios_dos = self.homogeneizar_puntos(self.vistas[0].features[self.vistas[1]].match_puntos_uno,self.vistas[0].features[self.vistas[1]].match_puntos_dos)
        self.obtener_camaras(puntos_homogenios_uno,puntos_homogenios_dos)
        self.triangular(puntos_homogenios_uno,puntos_homogenios_dos)

    def obtener_matriz_fundamental(self,puntos_clave_uno,puntos_clave_dos):
        self.F,self.FMask = cv2.findFundamentalMat(puntos_clave_uno,puntos_clave_dos,cv2.FM_RANSAC, 0.1, 0.99)

    def obtener_matriz_esencial(self):
        self.E = self.K.T.dot(self.F).dot(self.K)

    def enfrente_ambas_camaras(self,first_points, second_points, rot, trans):
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0]*rot[2, :],trans) / np.dot(rot[0, :] - second[0]*rot[2, :],second)
            first_3d_point = np.array([first[0] * first_z,second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,trans)
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
        return True

    def obtener_camaras(self,puntos_homogenios_uno,puntos_homogenios_dos):      
        U,s,V= np.linalg.svd(self.E,full_matrices=False)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,1.0]).reshape(3, 3)
        self.R = U.dot(W).dot(V) # U*W*V.T --> otra biblografia
        self.t = U[:,2]


        if not self.enfrente_ambas_camaras(puntos_homogenios_uno,puntos_homogenios_dos,self.R, self.t):
            # Second choice: R = U * W * Vt, T = -u_3
            self.t = - U[:, 2]
        if not self.enfrente_ambas_camaras(puntos_homogenios_uno,puntos_homogenios_dos,self.R, self.t):
            # Third choice: R = U * Wt * Vt, T = u_3
            self.R = U.dot(W.T).dot(V)
            self.t = U[:, 2]
            if not self.enfrente_ambas_camaras(puntos_homogenios_uno,puntos_homogenios_dos, self.R, self.t):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                self.t = - U[:, 2]
        self.camaras.append(np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.camaras.append(np.hstack((self.R, self.t.reshape(3, 1))))


    def triangulacion_lineal(u1, P1, u2, P2):
        A = np.array([u1[0]*P1[2, 0] - P1[0, 0], u1[0]*P1[2, 1] - P1[0, 1],
                      u1[0]*P1[2, 2] - P1[0, 2], u1[1]*P1[2, 0] - P1[1, 0],
                      u1[1]*P1[2, 1] - P1[1, 1], u1[1]*P1[2, 2] - P1[1, 2],
                      u2[0]*P2[2, 0] - P2[0, 0], u2[0]*P2[2, 1] - P2[0, 1],
                      u2[0]*P2[2, 2] - P2[0, 2], u2[1]*P2[2, 0] - P2[1, 0],
                      u2[1]*P2[2, 1] - P2[1, 1],
                      u2[1]*P2[2, 2] - P2[1, 2]]).reshape(4, 3)

        B = np.array([-(u1[0]*P1[2, 3] - P1[0, 3]),
                      -(u1[1]*P1[2, 3] - P1[1, 3]),
                      -(u2[0]*P2[2, 3] - P2[0, 3]),
                      -(u2[1]*P2[2, 3] - P2[1, 3])]).reshape(4, 1)

        ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        return X.reshape(1, 3)


    def homogeneizar_puntos(self,puntos_clave_uno,puntos_clave_dos):
        #cambiar esto esta mejor en el notebook
        
        puntos_homogenios_uno = []
        puntos_homogenios_dos = []
        for i in range(len(self.FMask)):
            if self.FMask[i]:
                # normalize and homogenize the image coordinates
                puntos_homogenios_uno.append(np.linalg.inv(self.K).dot([puntos_clave_uno[i][0],puntos_clave_uno[i][1], 1.0]))
                puntos_homogenios_dos.append( np.linalg.inv(self.K).dot([puntos_clave_dos[i][0],puntos_clave_dos[i][1], 1.0]))

        puntos_homogenios_uno = np.array(puntos_homogenios_uno)
        puntos_homogenios_dos = np.array(puntos_homogenios_dos)
        return puntos_homogenios_uno,puntos_homogenios_dos
        """
        unos = np.ones((puntos_clave_uno.shape[0],1))
        puntos_homogenios_uno = np.hstack((puntos_clave_uno,unos)) # normalizado a x,y,1
        puntos_homogenios_dos = np.hstack((puntos_clave_dos,unos)) # normalizado a x,y,1
        return puntos_homogenios_uno,puntos_homogenios_dos
        """


    def triangular(self,puntos_homogenios_uno,puntos_homogenios_dos):
        puntos_homogenios_uno = puntos_homogenios_uno.reshape(-1, 3)[:, :2]
        puntos_homogenios_dos = puntos_homogenios_dos.reshape(-1,3)[:,:2]
        triangulacion = cv2.triangulatePoints(self.camaras[0],self.camaras[1],puntos_homogenios_uno.T,puntos_homogenios_dos.T).T
        self.pts3D = triangulacion[:, :3]/np.repeat(triangulacion[:, 3], 3).reshape(-1, 3)
        return self.pts3D

    def graficar_nube(self):
        # plot with matplotlib
        Ys = self.pts3D[:, 0]
        Zs = self.pts3D[:, 1]
        Xs = self.pts3D[:, 2]
        # ------------------------->
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c='r', marker='o')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')

        plot.show()

    def salvar_nube(self,nombre):
        cloud = pcl.PointCloud(np.array(self.pts3D, dtype=np.float32))
        fil = cloud.make_statistical_outlier_filter()
        fil.set_mean_k (50)
        fil.set_std_dev_mul_thresh (1.0)
        fil.filter().to_file(nombre)