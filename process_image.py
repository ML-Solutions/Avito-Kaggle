import numpy as np
import pandas as pd
from time import time
import os
import tqdm
import re
import zipfile
from PIL import Image #pip install Pillow
import cv2 #pip install opencv-python
from skimage import feature
from collections import defaultdict
import operator
from scipy.stats import itemfreq
from multiprocessing import Pool
import matplotlib.pyplot as plt

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent
    
def perform_color_analysis(img):
    # cut the images into two halves as complete average may give bias results
    size = img.size
    halves = (size[0]/2, size[1]/2)
    im1 = img.crop((0, 0, size[0], halves[1]))
    im2 = img.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    return (light_percent, dark_percent)

def average_pixel_width(img): 
    im_array = np.asarray(img.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (img.size[0]*img.size[1]))
    return apw*100
    
def get_dominant_color(img):
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(arr.shape)

    dominant_color = palette[np.argmax(np.unique(labels,return_counts=True)[1])]
    return dominant_color[0]/255, dominant_color[1]/255, dominant_color[2]/255
    
def get_average_color(img):
    arr = np.float32(img)
    average_color = [arr[:, :, i].mean() for i in range(arr.shape[-1])]
    return average_color[0]/255, average_color[1]/255, average_color[2]/255
    
def getDimensions(img):
    return img.size 
    
def get_blurrness_score(img):
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
    return fm
    
def process_image(image_file):
    
    try:
        img = Image.open(image_file)
        img_size = os.stat(image_file).st_size
        lightness, darkness = perform_color_analysis(img)
        pixel_width = average_pixel_width(img)
        #dom_red, dom_green, dom_blue = get_dominant_color(img)
        avg_red, avg_green, avg_blue = get_average_color(img)
        width, height = getDimensions(img)
        blurness = get_blurrness_score(img)

        RE_CODE = re.compile(r'/(\w+)\.jpg')
        image_code = re.search(RE_CODE,image_file)[1]

        return [image_code, img_size, lightness, darkness, pixel_width, avg_red, avg_green, avg_blue, width, height, blurness]
    except:
        print('WARNING:',image_file,'cannot be open')
        RE_CODE = re.compile(r'/(\w+)\.jpg')
        image_code = re.search(RE_CODE,image_file)[1]
        return [image_code, 0, -999, -999, -999, -999, -999, -999, -999, -999, -999]
    #return [image_code, img_size, lightness, darkness, pixel_width, dom_red, dom_green, dom_blue, avg_red, avg_green, avg_blue, width, height, blurness]

if __name__ == "__main__":
    
    images = os.listdir("E:/Kaggle/Avito/images0/")
    images = ['E:/Kaggle/Avito/images0/'+image for image in images if image[-4:]=='.jpg']
    pool = Pool(4)
    #processed_images = pool.map(process_image,images)
    processed_images = list(tqdm.tqdm(pool.imap(process_image, images), total=len(images)))
    processed_images = pd.DataFrame(processed_images,columns=['image', 'img_size', 'lightness', 'darkness', 'pixel_width', 'avg_red', 'avg_green', 'avg_blue', 'width', 'height', 'blurness'])
    processed_images.to_csv('E:/Kaggle/Avito/image0_features.csv',index=False)
    pool.close()
    
    images = os.listdir("E:/Kaggle/Avito/images1/")
    images = ['E:/Kaggle/Avito/images1/'+image for image in images if image[-4:]=='.jpg']
    pool = Pool(4)
    processed_images = list(tqdm.tqdm(pool.imap(process_image, images), total=len(images)))
    processed_images = pd.DataFrame(processed_images,columns=['image', 'img_size', 'lightness', 'darkness', 'pixel_width', 'avg_red', 'avg_green', 'avg_blue', 'width', 'height', 'blurness'])
    processed_images.to_csv('E:/Kaggle/Avito/image1_features.csv',index=False)
    pool.close()
    
    images = os.listdir("E:/Kaggle/Avito/images2/")
    images = ['E:/Kaggle/Avito/images2/'+image for image in images if image[-4:]=='.jpg']
    pool = Pool(4)
    processed_images = list(tqdm.tqdm(pool.imap(process_image, images), total=len(images)))
    processed_images = pd.DataFrame(processed_images,columns=['image', 'img_size', 'lightness', 'darkness', 'pixel_width', 'avg_red', 'avg_green', 'avg_blue', 'width', 'height', 'blurness'])
    processed_images.to_csv('E:/Kaggle/Avito/image2_features.csv',index=False)
    pool.close()
    
    images = os.listdir("E:/Kaggle/Avito/images3/")
    images = ['E:/Kaggle/Avito/images3/'+image for image in images if image[-4:]=='.jpg']
    pool = Pool(4)
    processed_images = list(tqdm.tqdm(pool.imap(process_image, images), total=len(images)))
    processed_images = pd.DataFrame(processed_images,columns=['image', 'img_size', 'lightness', 'darkness', 'pixel_width', 'avg_red', 'avg_green', 'avg_blue', 'width', 'height', 'blurness'])
    processed_images.to_csv('E:/Kaggle/Avito/image3_features.csv',index=False)
    pool.close()
    
    images = os.listdir("E:/Kaggle/Avito/images4/")
    images = ['E:/Kaggle/Avito/images4/'+image for image in images if image[-4:]=='.jpg']
    pool = Pool(4)
    processed_images = list(tqdm.tqdm(pool.imap(process_image, images), total=len(images)))
    processed_images = pd.DataFrame(processed_images,columns=['image', 'img_size', 'lightness', 'darkness', 'pixel_width', 'avg_red', 'avg_green', 'avg_blue', 'width', 'height', 'blurness'])
    processed_images.to_csv('E:/Kaggle/Avito/image4_features.csv',index=False)
    pool.close()