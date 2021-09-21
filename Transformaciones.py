import numpy as np
import cv2 as cv
# import math as mt

###################### Listado de funciones para implementar las transformaciones #####################

# Normalizar una imagen, es decir, dejarla en un rango entre 0 y 1.
# Muchas transformaciones requeiren que la imagen de entrada esté en
# ese rango.
def Normalizar(Imagen):
    R, G, B = cv.split(Imagen)
    Imagen_Normalizada = cv.merge((R.astype('float')/255.0,
                                   G.astype('float')/255.0,
                                   B.astype('float')/255.0))
    return Imagen_Normalizada


# Comentar
def Pleno(Canal):
    return np.array(255*Canal, dtype=np.uint8)


# Comentar
def Escala(im):
  min_val=np.min(im.ravel())
  max_val=np.max(im.ravel())
  out = (im.astype('float') - min_val)/(max_val - min_val)
  return out


# Función que permite mapear los valores por encima de 255 y 0 a sus
# respectivos extremos.
def Limitar(im):
    Mayores = im > 255
    im[Mayores] = 255
    Menores = im < 0
    im[Menores] = 0
    return im


# Función base para realizar el balance de blancos
def imadjust(imagen, low_in, hig_in, low_out, hig_out, gamma):

    imagen_mod = low_out + (hig_out - low_out) * ((imagen - low_in) / (hig_in - low_in)) ** gamma
    imagen_mod = np.round(255 * imagen_mod)
    imagen_mod = Limitar(imagen_mod)
    imagen_mod = np.uint8(imagen_mod)
    return imagen_mod
#------------------------------------------------------------------------------------------------------

######################################## Transformaciones #############################################

# Normalización de la tinción
def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)


    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])


    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))

    Inorm[Inorm > 255] = 254

    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm

# Balance de Blancos
def white_balance(Imagen):
    R, G, B = cv.split(Imagen)

    Vec_R = R.astype('float') / 255.0
    Vec_G = G.astype('float') / 255.0
    Vec_B = B.astype('float') / 255.0

    Min_R = np.percentile(Vec_R, 1)
    Max_R = np.percentile(Vec_R, 99)

    Min_G = np.percentile(Vec_G, 1)
    Max_G = np.percentile(Vec_G, 99)

    Min_B = np.percentile(Vec_B, 1)
    Max_B = np.percentile(Vec_B, 99)

    DD1 = imadjust(Vec_R, Min_R, Max_R, 0.0, 1.0, 1)
    DD2 = imadjust(Vec_G, Min_G, Max_G, 0.0, 1.0, 1)
    DD3 = imadjust(Vec_B, Min_B, Max_B, 0.0, 1.0, 1)

    Imagen_Balance = cv.merge((DD1, DD2, DD3))

    return Imagen_Balance

# Ecualización del histograma
def histograma(img_in):

    # segregate color streams
    r, g, b = cv.split(img_in)

    equ_b = cv.equalizeHist(b)
    equ_g = cv.equalizeHist(g)
    equ_r = cv.equalizeHist(r)
    equ = cv.merge((equ_r, equ_g, equ_b))

    return equ

def histograma_canal(Imagen):

    ecualizacion = cv.equalizeHist(Imagen)

    return ecualizacion

# Histograma adaptativo
def histograma_adaptativo(Imagen):

    R, G, B = cv.split(Imagen)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    A = cv.merge((clahe.apply(R),
                  clahe.apply(G),
                  clahe.apply(B)))
    return A

# Ecualización Adaptativa + Balance de Blancos
def Balance_Adaptativo(Imagen):
    BB = white_balance(Imagen)
    AD = histograma_adaptativo(BB)

    return AD

# Canal (RGB): No se por qué, pero así es la única manera que Qt
# permite leer solo un canal en RGB. Solo en este espacio se hace la excepción.
def Canal_RGB(canal):

    imagen_mod = np.round(canal.astype('float'))
    imagen_mod = np.uint8(Limitar(imagen_mod))

    return imagen_mod

# Balance de Blancos sobre un solo canal
def balance_Bla_canal(Imagen):

    Vec_R = Imagen.astype('float') / 255.0

    Min_R = np.percentile(Vec_R, 1)
    Max_R = np.percentile(Vec_R, 99)

    DD1 = imadjust(Vec_R, Min_R, Max_R, 0.0, 1.0, 1)

    return DD1

# Histograma Adaptativo para un solo canal
def histograma_adaptativo_canal(Imagen):

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    Canal = clahe.apply(Imagen)

    return Canal

# Xyz + Balance de Blancos
def Xyz_balance(Imagen):

    Imagen = cv.cvtColor(Imagen, cv.COLOR_RGB2XYZ)
    Imagen = white_balance(Imagen)

    return Imagen

#Canal YCrCb
def Canal_YCrCb(Imagen, n):
    Imagen = cv.cvtColor(Imagen, cv.COLOR_RGB2YCrCb)
    if n == 0:
        Y = Imagen[:, :, 0]
        Y = balance_Bla_canal(Y)
        return Y
    elif n == 1:
        Cr = Imagen[:, :, 1]
        Cr = balance_Bla_canal(Cr)
        return Cr
    else:
        Cb = Imagen[:, :, 2]
        Cb = balance_Bla_canal(Cb)
        return Cb

# Canal Hearing
def OTHA(Imagen, n):

    R = Imagen[0:, 0:, 0]/255.0
    G = Imagen[0:, 0:, 1]/255.0
    B = Imagen[0:, 0:, 2]/255.0

    if n == 0:
        I1 = (R + G + B)/3
        I1 = Escala(I1)
        I1 = np.array(255 * I1, dtype=np.uint8)
        I1 = balance_Bla_canal(I1)
        return I1
    elif n == 1:
        I2 = (R - B) / 2.0
        I2 = Escala(I2)
        I2 = np.array(255 * I2, dtype=np.uint8)
        I2 = balance_Bla_canal(I2)
        return I2
    else:
        I3 = (2*G - R - B)/4
        I3 = Escala(I3)
        I3 = np.array(255 * I3, dtype=np.uint8)
        return I3

# CMYK


# Formula to convert RGB to CMY.
def rgb_to_cmy(r, g, b):
    # RGB values are divided by 255
    # to bring them between 0 to 1.
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255
    return c, m, y

def CMY(Imagen, n):

    R = Imagen[0:, 0:, 0]/255.0
    G = Imagen[0:, 0:, 1]/255.0
    B = Imagen[0:, 0:, 2]/255.0

    c = 1 - R
    m = 1 - G
    y = 1 - B

    if n == 0:
        c = np.round(255*c)
        c = Escala(c)
        c = balance_Bla_canal(c)
        return c
    elif n == 1:
        m = np.round(255*m)
        m = Escala(m)
        m = balance_Bla_canal(m)
        return m
    else:
        y = np.round(255*y)
        y = Escala(y)
        y = balance_Bla_canal(y)
        return y

# YIQ
def YIQ(Imagen, n):

    R, G, B = cv.split(Imagen)

    Y = 0.299*R + 0.587*G + 0.114*B
    I = 0.596*R - 0.275*G - 0.321*B
    Q = 0.212*R - 0.523*G + 0.311*B

    if n == 0:
        Y = Escala(Y)
        Y = Pleno(Y)
        Y = balance_Bla_canal(Y)
        return Y
    elif n == 1:
        I = Escala(I)
        I = Pleno(I)
        I = balance_Bla_canal(I)
        return I
    else:
        Q = Escala(Q)
        Q = Pleno(Q)
        return Q

# YUV
def YUV(Imagen, n):

    Imagen = cv.cvtColor(Imagen, cv.COLOR_RGB2YUV)

    Y = Imagen[0:,0:,0]
    U = Imagen[0:,0:,1]
    V = Imagen[0:,0:,2]
    if n == 0:
        U = balance_Bla_canal(U)
        return U
    else:
        V = balance_Bla_canal(V)
        return V



def Lab(Imagen, n):
    Lab = cv.cvtColor(Imagen, cv.COLOR_RGB2LAB)
    L, a, b = cv.split(Lab)
    if n == 0:
        L = balance_Bla_canal(L)
        return L
    elif n == 1:
        a = balance_Bla_canal(a)
        return a
    else:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b = balance_Bla_canal(b)
        b = clahe.apply(b)
        return b

# Luv
def Luv(Imagen, n):
    Imagen = cv.cvtColor(Imagen, cv.COLOR_RGB2Luv)
    L, u, v = cv.split(Imagen)
    if n == 0:
        u = balance_Bla_canal(u)
        return u
    else:
        v = balance_Bla_canal(v)
        return v

# HSI
def HSI(Imagen, n):
    Imagen = cv.cvtColor(Imagen, cv.COLOR_RGB2HSV)
    H, S, V = cv.split(Imagen)
    if n == 0:
        H = balance_Bla_canal(H)
        return H
    else:
        S = balance_Bla_canal(S)
        return S

# HMMD
def HMMD(Imagen, n):

    R, G, B = cv.split(Imagen)
    if n == 0:
        Ma = np.maximum(np.maximum(R, G), B)
        Ma = balance_Bla_canal(Ma)
        return Ma
    elif n == 1:
        Mi = np.minimum(np.minimum(R, G), B)
        Mi = balance_Bla_canal(Mi)
        return Mi
    else:
        Mi = np.minimum(np.minimum(R, G), B)
        Mi = balance_Bla_canal(Mi)
        Ma = np.maximum(np.maximum(R, G), B)
        Ma = balance_Bla_canal(Ma)
        D = Ma - Mi
        D = balance_Bla_canal(D)
        return D

