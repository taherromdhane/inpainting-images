import dash_html_components as html
from dash_canvas.utils import parse_jsonstring

import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import sys
import time

# Utility function to save contents to memory
def saveImage(contents, filepath):
    
    data = contents.encode("utf8").split(b";base64,")[1]

    with open(os.getcwd() + filepath, "wb") as fp:
        fp.write(base64.decodebytes(data))  


# Utility function to show preview of uploaded file
def parseContents(contents, filename):
    return html.Div([
        #html.H5(filename),
        html.Img(
            src = contents,
            style = {
                'width' : '100%',
                'height' : 'auto'
            },
            className = 'preview-image'
        ),
    ])

def parseContentsDir(source, filename):
    return html.Div([
        #html.H5(filename),
        html.Img(
            src = source,
            style = {
                'width' : '100%',
                'height' : 'auto'
            },
            className = 'preview-image'
        ),
    ])

def reduceImageSize(source, max_dimension) :
    im = Image.open(source)
    width, height = im.size

    ratio = max(width / max_dimension, height / max_dimension)
    if ratio > 1 :
        im = im.resize((int(width / ratio), int(height / ratio)))
        im.save(source)

# Utility function to get rectangular mask from dash canvas
def getMask(image_width, image_height, left, top, width, height) :

    mask = np.ones((image_height, image_width))
    mask[top : top + height + 1, left : left + width + 1] = 0

    return mask

def readMask(mask_filepath) :

    mask = np.array(Image.open(mask_filepath))
    if len(mask.shape) >= 3 :
        mask = 1 * (mask == np.min(mask))[:, :, 0]

    return mask

# Utility function to get mask from data of dash canvas
def maskFromData(string, data, mask_filepath, image_width, image_height, canvas_width, rect_fill) :
    mask = np.zeros((image_height, image_width))

    if 'fill' in rect_fill : 
        for object in data['objects'] :
            if object["type"] == "rect" :
                    
                left = int(object["left"] * image_width / canvas_width)
                top = int(object["top"] * image_width / canvas_width)
                width = int(object["width"] * image_width / canvas_width)
                height = int(object["height"] * image_width / canvas_width)

                rect_mask = getMask(image_width, image_height, left, top, width, height)
                mask[rect_mask == 0] = 1
    
    mask[parse_jsonstring(string, (image_height, image_width)) == 1] = 1

    plt.imsave(os.getcwd() + mask_filepath, mask, cmap=cm.gray)

# Utility function to run inpainter and save results priodically

from inpaint.Inpainter import Inpainter

def inpaintingOneIterationLogic(
        image, 
        mask, 
        patch_size, 
        local_radius, 
        data_significance, 
        threshold, 
        live_update, 
        inpainted_filepath, 
        mask_filepath, 
        progress_filepath
    ) :

    inpainter = Inpainter(patch_size, local_radius = local_radius, data_significance = data_significance, threshold = threshold)

    inpainted, new_mask, progress = inpainter.inpaintOneIteration(image, mask)

    # Update progress
    with open(progress_filepath, 'w') as fp :
        fp.write(str(progress))

    if live_update :
        # Update image
        plt.imsave(os.getcwd() + inpainted_filepath, inpainted)
        plt.imsave(os.getcwd() + mask_filepath, new_mask)

def inpaintingLogic(
        image, 
        mask, 
        patch_size, 
        local_radius, 
        data_significance, 
        threshold, 
        live_update, 
        inpainted_filepath, 
        progress_filepath
    ) :

    inpainter = Inpainter(patch_size, local_radius = local_radius, data_significance = data_significance, threshold = threshold)

    start_time = time.time()
    seconds_passed = 0
    inpainted = None
    for inpainted, progress, mask in inpainter.inpaintWithSteps(image, mask) :

        # Update progress
        with open(os.getcwd() + progress_filepath, 'w') as fp :
            fp.write(str(progress))

        if live_update :
            # Update image every 1s
            elapsed = int(time.time() - start_time)

            if elapsed > seconds_passed :
                inpainted[mask == 0] = 0
                plt.imsave(os.getcwd() + inpainted_filepath, inpainted)
                seconds_passed = elapsed
                # print("elapsed : ", elapsed)

    plt.imsave(os.getcwd() + inpainted_filepath, inpainted)