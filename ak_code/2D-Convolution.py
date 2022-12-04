import cv2
import numpy as np


I = int(input("'1' for wolves.png \n '2' for lena.png"))
if I == 1:
    testImage = cv2.imread(r'D:\NCSU\NCSU Courses\DIS - 558\DIS Project2\wolves.png')
    grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    imageShape = grayImage.shape
    cv2.imshow('Original',grayImage)
    cv2.waitKey(0)
else:
    testImage = cv2.imread(r'D:\NCSU\NCSU Courses\DIS - 558\DIS Project2\lena.png')
    grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    imageShape = grayImage.shape
    cv2.imshow('Original',grayImage)
    cv2.waitKey(0)

#AVERAGE
def avgFilter():
  krnl = np.ones((3,3))/(3**2)
  return krnl

#PREWITT
def prewittFilter_mx():
  krnl = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
  return krnl

def prewittFilter_my():
  krnl = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])
  return krnl

#SOBEL
def sobelFilter_mx():
  krnl = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
  return krnl

def sobelFilter_my():
  krnl = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
  return krnl

#ROBERTS
def robertsFilter_mx():
  krnl = np.array([[0, 1],
                   [-1, 0]])
  return krnl

def robertsFilter_my():
  krnl = np.array([[1, 0],
                   [0, -1]])
  return krnl

#FIRST ORDER DERIVATIVE
def firstOrder():
    fod = int(input("'1' for first order row matrix \n '2' for first order column matrix"))
    if fod == 1:
        krnl = np.array([[-1, 1]])
    elif fod == 2:
        fod_c = int(input("'1' for column type 1 -- [-1;1] \n '2' for column type 2 -- [1;-1]"))
        if fod_c == 1:
            krnl = np.array([[-1],[1]])
        elif fod_c == 2:
            krnl = np.array([[1], [-1]])
    return krnl

#ZERO PADDING
def zeroPadding(f, w):
    size = f.shape

    horizontalPadding = size[0] + w - 1
    verticalPadding = size[1] + w - 1
    paddedImage = np.zeros ((horizontalPadding,verticalPadding))

    for r in range(size[0]):
      for c in range(size[1]):
        paddedImage[r+int((w-1)/2),c+int((w-1)/2)] = grayImage[r,c]
    return paddedImage

#WRAP AROUND
def wrapAround(f,w):
    image = f

    paddingMeausre = w
    leftTop=f[: paddingMeausre,: paddingMeausre]
    leftBottom=f[-paddingMeausre :,: paddingMeausre]
    rightTop=f[: paddingMeausre,-paddingMeausre :]
    rightBottom=f[-paddingMeausre :,-paddingMeausre :]

    left = np.array([f[:,-1]]).T
    right = np.array([f[:, 0]]).T
    for i in range(paddingMeausre):
            image = np.vstack(((f[-1-i,:]) , image, (f[(0+i),:])))
            if i < (paddingMeausre-1):
                left = np.hstack(((np.array([f[:,-(2+i)]]).T), left))
                right = np.hstack((right,(np.array([f[:,(1+i)]]).T)))
            else:
                left = np.vstack((rightBottom, left, rightTop))
                right = np.vstack((leftBottom, right, leftTop))

    image= np.hstack(((left, image, right)))
    return image

#COPY EDGE
def copyEdge(f,w):
    image = f
    paddingMeausre = w
    leftTop = f[: paddingMeausre, : paddingMeausre]
    leftBottom = f[-paddingMeausre:, : paddingMeausre]
    rightTop = f[: paddingMeausre, -paddingMeausre:]
    rightBottom = f[-paddingMeausre:, -paddingMeausre:]

    left = np.array([f[:,-1]]).T
    right = np.array([f[:, 0]]).T
    for i in range(paddingMeausre):
            image = np.vstack(((f[0,:]) , image, (f[-1,:])))
            if i < (paddingMeausre-1):
                left = np.hstack(((np.array([f[:,0]]).T), left))
                right = np.hstack((right,(np.array([f[:,-1]]).T)))
            else:
                left = np.vstack((leftTop , left , leftBottom))
                right = np.vstack((rightTop , right , rightBottom))

    image= np.hstack(((left, image, right)))
    return image

#REFLECT ACROSS
def reflectAcross(f,w):
    image = f
    paddingMeausre = w
    for i in range (paddingMeausre):
        image = np.vstack(((f[(1+i),:]) , image, (f[-(2+i),:])))
        outputImage = image
    for i in range (paddingMeausre):
        outputImage = np.hstack(((np.array([image[:,(1+i)]]).T), outputImage, (np.array([image[:,-(2+i)]]).T)))
    return outputImage

#PADDING
def padding_Type1(img,pT,w):
    if pT == 1:
        paddedImage = zeroPadding(img, w)
        d = conv2Gray(paddedImage, kernal)
    elif pT == 2:
        paddedImage = wrapAround(img, w)
        d = conv2Gray(paddedImage, kernal)
    elif pT == 3:
        paddedImage = copyEdge(img, w)
        d = conv2Gray(paddedImage, kernal)
    elif pT == 4:
        paddedImage = reflectAcross(img, w)
        d = conv2Gray(paddedImage, kernal)
    return d

def padding_Type2(img,pT,w):
    if pT == 1:
        paddedImage = zeroPadding(img, w)

    elif pT == 2:
        paddedImage = wrapAround(img, w)

    elif pT == 3:
        paddedImage = copyEdge(img, w)

    elif pT == 4:
        paddedImage = reflectAcross(img, w)

    return paddedImage
#2D CONVOLUTION

def conv2Gray(f, w):
    paddedWidth, paddedHeight = f.shape
    kernalWidth, kernalHeight = w.shape
    stride = int(input("stride:"))
    newWidth = (paddedWidth - kernalWidth) // stride +1
    newHeight = (paddedHeight - kernalHeight) // stride +1

    newImage = np.zeros((newWidth, newHeight)).astype(np.float32)
    for x in range(newWidth):
        if x > paddedWidth - kernalWidth:
            break
        if x % stride == 0:
            for y in range(newHeight):
                if y > paddedHeight - kernalHeight:
                    break
                if y % stride == 0:
                    newImage[x][y] = np.sum(f[x * stride:x * stride + kernalWidth, y * stride:y * stride +kernalHeight]*kernal).astype(np.float32)
    return  newImage

def conv2RGB(f, paddingType,kernal, w):
    b,g,r = cv2.split(f)
    b = np.array(b)
    g = np.array(g)
    r = np.array(r)
    b = padding_Type2(b,paddingType,w)
    g = padding_Type2(g, paddingType, w)
    r = padding_Type2(r, paddingType, w)
    paddedWidth, paddedHeight = b.shape
    kernalWidth ,kernalHeight = kernal.shape

    stride = int(input("stride:"))
    newWidth = (paddedWidth - kernalWidth) // stride +1
    newHeight = (paddedHeight - kernalHeight) // stride +1
    newImage_b= np.zeros((newWidth, newHeight)).astype(np.float32)
    newImage_g = np.zeros((newWidth, newHeight)).astype(np.float32)
    newImage_r = np.zeros((newWidth, newHeight)).astype(np.float32)

    for x in range(newWidth):
        if x > paddedWidth - kernalWidth:
            break
        if x % stride == 0:
            for y in range(newHeight):
                if y > paddedHeight - kernalHeight:
                    break
                if y % stride == 0:
                    newImage_b[x][y] = np.sum(b[x * stride:x * stride + kernalWidth, y * stride:y * stride +kernalHeight]*kernal).astype(np.float32)
                    newImage_g[x][y] = np.sum(g[x * stride:x * stride + kernalWidth, y * stride:y * stride + kernalHeight] * kernal).astype(np.float32)
                    newImage_r[x][y] = np.sum(r[x * stride:x * stride + kernalWidth, y * stride:y * stride + kernalHeight] * kernal).astype(np.float32)

    newImage = cv2.merge([newImage_r,newImage_g,newImage_b])
    return newImage




kernalType = int(input("'1' for averaging filter \n '2' for Prewitt Filter \n '3' for Sobel Filter \n '4' for Roberts Filter \n '5' for first order derivative Filters"))
if kernalType == 1:
    kernal = avgFilter()
elif kernalType == 2:
    type = input("mx or my:")
    if type =="mx":
        kernal = prewittFilter_mx()
    else:
        kernal = prewittFilter_my()
elif kernalType == 3:
    type = input("mx or my:")
    if type == "mx":
        kernal = sobelFilter_mx()
    else:
        kernal = sobelFilter_my()
elif kernalType == 4:
    type = input("mx or my:")
    if type == "mx":
        kernal = robertsFilter_mx()
    else:
        kernal = robertsFilter_my()
elif kernalType == 5:
    kernal = firstOrder()

paddingType=int(input("'1' for zero padding \n '2' for wrap around \n '3' for copy edge \n '4' for reflect across edge"))
w = int(input("Padding Dimension:"))
imageType = int(input("'1' for Gray Scale \n '2' for RGB image"))
if imageType == 1:
    d = padding_Type1(grayImage,paddingType,w)
else:
    d = conv2RGB(testImage,paddingType,kernal,w)

convolutedImage = d[w:d.shape[0]-w, w:d.shape[1]-w]

cv2.imwrite('paddedImage.png', d)
testImage1 = cv2.imread(r'D:\PyCharm Community Edition 2022.2.2\pythonProject2\paddedImage.png')
cv2.imshow('Padded Image',testImage1)
cv2.waitKey(0)
cv2.imwrite('convoluted.png', convolutedImage)
testImage2 = cv2.imread(r'D:\PyCharm Community Edition 2022.2.2\pythonProject2\convoluted.png')
cv2.imshow('Convoluted Image',testImage2)
cv2.waitKey(0)
cv2.destroyAllWindows()