import os
import cv2
import glob as gb
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

img_src_path = "E:/processing/1精子/精子形态参考"
all_imgs = gb.glob("E:/processing/1精子/精子形态参考/*.png")


#all_imgs = os.listdir(img_src_path)
#print(str(all_imgs))
# def get_images(img_src_path):
#    all_imgs = os.listdir(img_src_path)
CROP_HEIGHT_BEGIN = 0
CROP_HEIGHT = 1040
CROP_WIDTH_BEGIN = 0
CROP_WIDTH = 1388


def cv2pil(input_img):
    pil_image = Image.fromarray(cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB))
    return pil_image

def pil2cv(input_image):
    cv_img = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
    return cv_img


def cropping(input_img, cropheightbegin, cropheight, cropwidthbegin, cropwidth):
    cropped_img = input_img[cropheightbegin:cropheightbegin+cropheight, cropwidthbegin:cropwidthbegin+cropwidth]
    return cropped_img

def reshape(input_img, cropheight, cropwidth):
    reshapped = cv2.resize(input_img, (500, 500), interpolation=cv2.INTER_CUBIC)
    return reshapped


def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = cdf.astype(np.uint8)
    img3 = cdf[img]
    return img3

def image_enhance(input_img, mode):
    if mode == 0:     # 0 means histogram equalization
        enhanced_img = histogram_equalization(input_img)      # Histogram euqalization

    elif mode == 1:     # 1 means Laplace Enhance
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced_img = cv2.filter2D(input_img, cv2.CV_8UC3, kernel)      # Laplace Enhance

    elif mode == 2:
        enhanced_img = np.uint8(np.log(np.array(input_img) + 1))
        cv2.normalize(enhanced_img, enhanced_img, 0, 255, cv2.NORM_MINMAX)

    elif mode == 3:
        fgamma = 2
        enhanced_img = np.uint8(np.power((np.array(input_img) / 255.0), fgamma) * 255.0)
        cv2.normalize(enhanced_img, enhanced_img, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(enhanced_img, enhanced_img)
    #cv2.imshow("enhanced", enhanced_img)
    return enhanced_img


def blur_operations(input_img, blur_mode):
    if blur_mode == 0:      # Median filter
        blurred = cv2.medianBlur(input_img, 5)
    elif blur_mode == 1:    # Bilateral filter
        blurred = np.hstack([  #cv2.bilateralFilter(input_img, 5, 21, 21)
            cv2.bilateralFilter(input_img, 7, 31, 31),
            # cv2.bilateralFilter(gray_img, 9, 41, 41)
        ])

    #cv2.imshow("blurred img", blurred)
    return blurred

def color2gray(input_img):
    # cropped = cropping(img, CROP_HEIGHT_BEGIN, CROP_HEIGHT, CROP_WIDTH_BEGIN, CROP_WIDTH)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)    # Convert the bgr image to a gray scale one
    cv2.imshow("grayimg", gray_img)
    return gray_img


def img_preprocess(img):
    enhanced_img = image_enhance(img, 3)  # Histogram equalization
    enhanced_img = image_enhance(enhanced_img, 0)
    blurred_gray_img=cv2.GaussianBlur(enhanced_img,(5,5),5)
    blurred_gray_img=cv2.GaussianBlur(blurred_gray_img,(5,5),5)
    #blurred_gray_img = blur_operations(enhanced_img, 0)
    
    cv2.imshow("blurring", blurred_gray_img)

    preprocessed_img = blurred_gray_img
    return preprocessed_img


def binary_convert(input_img):
    #bina_inv_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, bina_img = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bina_inv_img = cv2.bitwise_not(bina_img)
    cv2.imshow("binary", bina_inv_img)
    return bina_inv_img

def binary2color(input_binary_img):
    canvas = np.zeros((int(CROP_HEIGHT * 0.6), int(CROP_WIDTH * 0.6), 3), dtype="uint8")
    mask = np.where(input_binary_img == 255)
    for r in range(0, np.size(mask, 1)):
        canvas[mask[0][r]][mask[1][r]][0] = 255
        canvas[mask[0][r]][mask[1][r]][1] = 255
        canvas[mask[0][r]][mask[1][r]][2] = 255
    return canvas

def morphology_process(input_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10,10))
    opened = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernel3)
    #closed = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)

    #eroded = cv2.erode(closed, kernel)
    #eroded = cv2.erode(eroded, kernel3)
    #dilated = cv2.dilate(opened, kernel)
    eroded = cv2.erode(opened, kernel)
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel3)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
    #dilated = cv2.dilate(closed, kernel)
    #closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel2)
    morph_img = closed
    cv2.imshow("morphology", morph_img)
    return morph_img


def img_findContours(input_img):
    '''
    # Finding Contours
    img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours, np.shape(contours))
    # cnt = contours[0]
    cv2.drawContours(img2, contours, -1, (0, 255, 0), 3)
    area = cv2.contourArea(contours)
    print(area)
    '''
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(input_img)

    print(stats[1:-1])
    # print(centroids)
    return stats[1:-1]


def drawComponets(input_img, comp):
    shape = np.shape(comp)
    for i in range(0, shape[0]):
        if comp[i][4]>100 and comp[i][4]<1000:
            cv2.rectangle(input_img, (comp[i][0]-10, comp[i][1]-10), (comp[i][0]+comp[i][2]+10, comp[i][1]+comp[i][3]+10), (255, 255, 0), 2)
    cv2.imshow("found_components", input_img)


def hough_circle(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                20, param1=100, param2=30, minRadius=20, maxRadius=70)
    if circles1 is None:
        print("Didn't find any circles in the picture")
    else:
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(input_img, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv2.circle(input_img, (i[0], i[1]), 2, (255, 0, 255), 10)
            cv2.rectangle(input_img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)

        print("Coodinate of center: (", i[0], ",", i[1], "), Radius: ", i[2])
        plt.figure()
        plt.imshow(input_img)
        plt.xticks([]), plt.yticks([])
        plt.show()
import tensorflow as tf
tf.reset_default_graph()
with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('E:/processing/1精子/CNN/model.ckpt.meta')
    new_saver.restore(sess,'E:/processing/1精子/CNN/model.ckpt')
    graph = tf.get_default_graph()
    x=graph.get_operation_by_name('x_1').outputs[0]
    prediction=tf.get_collection("pred_network")[-1]
    for i in all_imgs:
        os.chdir('E:/processing/1精子/精子形态参考')
        each_image=i
        each_image=each_image[25:]
        img_mat = cv2.imread(each_image)
        reshaped = reshape(img_mat, CROP_HEIGHT, CROP_WIDTH)
        gray_img = color2gray(reshaped)
        gray_img_2=gray_img.copy()
        pre_img = img_preprocess(gray_img)
        binary_img = binary_convert(pre_img) 
        morphology = morphology_process(binary_img)
        binary_img=morphology
        components = img_findContours(binary_img)
        #保存框框
        components=[i for i in components if i[4]>100 and i[4]<1000]
        comp=[]
        w,l=500,500
        for j in range(len(components)):
            kk=components[j].copy()
            for k in range(2):
                if kk[k]>20:kk[k]=kk[k]-20
                else:kk[k]=0
            if kk[1]+kk[3]+40>=w:ww=w
            else:ww=int(kk[1]+kk[3]+40)
            if kk[0]+kk[2]+40>l:ll=l
            else:ll=int(kk[0]+kk[2]+40)
            cropImg = reshaped[kk[1]:ww,kk[0]:ll]
            cropImg = cv2.resize(cropImg, (64,64), interpolation=cv2.INTER_CUBIC)
            cropImg = color2gray(cropImg)
            cropImg= binary_convert(cropImg)/255
            cropImg= np.expand_dims(cropImg, 2)
            comp.append(cropImg)
            cv2.destroyAllWindows() 
        comp=np.array(comp)
        pre=sess.run(prediction,feed_dict={x: comp})
        need=[cc for cc in range(len(pre)) if pre[cc][1]>0.5]
        print(pre)
        components=np.array(components  )
        components=components[need]
        drawComponets(gray_img, components)
        cv2.waitKey()
        cv2.destroyAllWindows() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

















