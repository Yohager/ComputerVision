# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import cv2
import time
import math
import warnings

warnings.filterwarnings("ignore")

def my_imfilter(image, filter):
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
  #print('原图的尺寸', image.shape)
  #print('卷积核的尺寸', filter.shape)
  '''
  :param image:image是一个三维的矩阵：m*n*c，其中m*n是单个通道的尺寸，c是通道的数量
  :param filter:卷积核的尺寸k*l
  :return:返回一个卷积后的feature map
  '''
  # 这里处理的卷积都是输出原尺寸，即先对原图进行pad操作，然后再卷积
  # 首先进行padding
  image_padding = np.pad(image, (
  ((filter.shape[0] - 1) // 2, (filter.shape[0] - 1) // 2), ((filter.shape[1] - 1) // 2, (filter.shape[1] - 1) // 2),
  (0, 0)), 'constant')
  #print('padding之后图像的尺寸', image_padding.shape)
  # 计算输出图像的尺寸
  output_image_height = image_padding.shape[0] - filter.shape[0] + 1
  output_image_width = image_padding.shape[1] - filter.shape[1] + 1
  output_image = np.zeros([output_image_height, output_image_width, image.shape[2]])
  time_start = time.clock()
  for k in range(image.shape[2]):
    for i in range(output_image_height):
      for j in range(output_image_width):
        output_image[i, j, k] = np.sum(
          np.multiply(image_padding[i:i + filter.shape[0], j:j + filter.shape[1], k], filter))
  time_finish = time.clock()
  #print('卷积用时为：', time_finish - time_start)
  #print('卷积之后的图像尺寸', output_image.shape)
  return output_image

def clamp(pixel_value):
    if pixel_value >= 255:
        return 255
    if pixel_value <= 0:
        return 0
    else:
        return pixel_value

def gen_hybrid_image(image1, image2, cutoff_frequency):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """
  #断言：判断两个输入进来的图片的尺寸大小是否一样
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
  s, k = cutoff_frequency, cutoff_frequency*2
  probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
  kernel = np.outer(probs, probs)
  #print('卷积核的显示',kernel)
  #print('卷积核的尺寸大小：',kernel.shape)
  # Your code here:
  low_frequencies_1 = cv2.filter2D(src=image1,ddepth=-1,kernel=kernel)
  #print("cv2的低通滤波器得到的图像尺寸",low_frequencies_1.shape)
  low_frequencies = my_imfilter(image1,kernel)
  low_frequencies = low_frequencies.astype(np.uint8)
  #print("低通滤波器得到的图像尺寸",low_frequencies.shape)
  cv2.imshow('low_frequencies',low_frequencies)

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  #high_frequencies = image2 - cv2.filter2D(src=image2,ddepth=-1,kernel=kernel)
  high_frequencies = image2 - my_imfilter(image2,kernel)
  high_frequencies = high_frequencies.astype(np.uint8)
  #print("高通滤波器得到的图像尺寸",high_frequencies.shape)
  cv2.imshow('high_frequencies',high_frequencies)
  # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = low_frequencies + high_frequencies # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  for k in range(hybrid_image.shape[0]):
    for i in range (hybrid_image.shape[1]):
      for j in range (hybrid_image.shape[2]):
        hybrid_image[k,i,j] = clamp(hybrid_image[k,i,j])
  cv2.imshow('result',hybrid_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  hybrid_image = hybrid_image/255.0
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))


if __name__ == '__main__':
  filepath_1 = '..\data\dog.bmp'
  filepath_2 = '..\data\cat.bmp'
  image_1 = cv2.imread(filepath_1)
  image_2 = cv2.imread(filepath_2)
  #print(image_1.shape)
  low,high,merge = gen_hybrid_image(image_1,image_2,5)
  output_pic = vis_hybrid_image(merge)
  #print(output_pic.shape)
  #output_pic = output_pic.astype(np.uint8)

  cv2.imshow('downsampling',output_pic)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #print('低频图像：',low)
  #filter = np.ones([3,3])
  #filtered_image = my_imfilter(image_1,filter)
  #print(filtered_image.shape)
