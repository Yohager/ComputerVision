# --*-- coding:utf-8 --*--
'''
author: yohager
date: 2019.6.13
task: computer vision-filtering
'''
import cv2
import numpy as np


#均值滤波
def average_blur(image):
    filter_average = cv2.blur(image,(1,15))
    cv2.imshow("average_blur",filter_average)

#中值滤波
def median_blur(image):
    filter_median = cv2.medianBlur(image,5)
    cv2.imshow("median_blur",image)


#防止像素点的值溢出0-255的范围
def clamp(pixel_value):
    if pixel_value >= 255:
        return 255
    if pixel_value <= 0:
        return 0
    else:
        return pixel_value

#高斯滤波
def gaussian_blur(image):
    filter_gaussian = cv2.GaussianBlur(image,(19,19),0)
    cv2.imshow("Gaussian_Blur",filter_gaussian)

#高通滤波
def sobel_blur(image):
    filter_sobel = cv2.Sobel(image,cv2.CV_16S,0,1)
    image_result = cv2.convertScaleAbs(filter_sobel)
    cv2.imshow('Sobel_Blur',image_result)

if __name__ == '__main__':
    filepath_1 = '..\data\dog.bmp'
    filepath_2 = '..\data\cat.bmp'
    image_1 = cv2.imread(filepath_1)
    image_2 = cv2.imread(filepath_2)
    cv2.imshow('init_image_dog',image_1)
    cv2.imshow('init_image_cat',image_2)
    gaussian_blur(image_1)
    sobel_blur(image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()