# --*-- coding:utf-8 --*--
'''
author: yohager
date: 2019.6.14
task: try to fasten conv
'''

import numpy as np
import datetime
import time
import cv2
import math


#暂时默认所有的stride都是1
def common_conv(image,filter):
    print('原图的尺寸',image.shape)
    print('卷积核的尺寸',filter.shape)
    '''
    :param image:image是一个三维的矩阵：m*n*c，其中m*n是单个通道的尺寸，c是通道的数量
    :param filter:卷积核的尺寸k*l
    :return:返回一个卷积后的feature map
    '''
    #这里处理的卷积都是输出原尺寸，即先对原图进行pad操作，然后再卷积
    #首先进行padding
    image_padding = np.pad(image,(((filter.shape[0]-1)//2,(filter.shape[0]-1)//2),((filter.shape[1]-1)//2,(filter.shape[1]-1)//2),(0,0)),'constant')
    print('padding之后图像的尺寸',image_padding.shape)
    #计算输出图像的尺寸
    output_image_height = image_padding.shape[0] - filter.shape[0] + 1
    output_image_width = image_padding.shape[1] - filter.shape[1] + 1
    output_image = np.zeros([output_image_height,output_image_width,image.shape[2]])
    time_start = time.clock()
    for k in range(image.shape[2]):
        for i in range(output_image_height):
            for j in range(output_image_width):
                output_image[i,j,k] = np.sum(np.multiply(image_padding[i:i+filter.shape[0],j:j+filter.shape[1],k],filter))
    time_finish = time.clock()
    print('卷积用时为：',time_finish-time_start)
    print('卷积之后的图像尺寸',output_image.shape)
    output_image = output_image.astype(np.uint8)
    return output_image


def my_imfilter_1(image, filter):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter to an image. Return the filtered image.
    Inputs:
    - image -> numpy nd-array of dim (m, n, c)
    - filter -> numpy nd-array of odd dim (k, l)
    Returns
    - filtered_image -> numpy nd-array of dim (m, n, c)
    Errors if:
    - filter has any even dimension -> raise an Exception with a suitable error message.
    """
    filter_reshape = filter.reshape(-1,1)
    filtered_image = np.zeros(image.shape)
    filter_size = filter.size
    filter_size_sqrt = round(math.sqrt(filter_size))
    filter_size_ceil = math.ceil(filter_size_sqrt/2)
    filter_size_floor = math.floor(filter_size_sqrt/2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i>=filter_size_floor and i<=image.shape[0]-filter_size_ceil and j>=filter_size_floor and j<=image.shape[1]-filter_size_ceil:
                convolute_image = image[i-filter_size_floor:i+filter_size_ceil, j-filter_size_floor:j+filter_size_ceil]
                reshape_image = np.reshape(convolute_image[0:filter_size_sqrt,0:filter_size_sqrt], (filter_size,3))
                for k in range(filter_reshape.shape[0]):
                    filtered_image[i,j] += filter_reshape[k] * reshape_image[k]
    return filtered_image

if __name__ == '__main__':
    filpath = '..\data\cat.bmp'
    image = cv2.imread(filpath)
    print('输入图像的尺寸',image.shape)
    filter = np.ones([5,5]) /25.0
    #print(filter.shape)
    test_image = np.ones([5,5,3])
    result_1 = cv2.filter2D(image,-1,filter)
    result_2 = common_conv(image,filter)
    cv2.imshow('init',image)
    cv2.imshow('after_filter_1',result_1)
    cv2.imshow('after_filter_2',result_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()