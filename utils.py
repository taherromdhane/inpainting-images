import dash_html_components as html
from dash_canvas.utils import parse_jsonstring
from inpaint.Inpainter import Inpainter

import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import time

def saveImage(contents, filepath):
    """
        Utility function to save contents of an image (encoded in base 64)
        to memory (file)
        used to save uploaded file to server
        Parameters :
            contents: contents of the image in binary base 64 format
            filepath: path to save the image to
    """
    
    data = contents.encode("utf8").split(b";base64,")[1]

    with open(os.getcwd() + filepath, "wb") as fp:
        fp.write(base64.decodebytes(data))  


# Utility function to show preview of uploaded file
def parseContents(contents, filename):
    """
        Utility function to parse the contents of the image (in base 64 format) 
        and return an element where it is displayed
        Parameters :
            contents: contents of the image in binary base 64 format
            filepath: name of the image
    """

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
    """
        Utility function to parse the contents of the image from a file 
        and return an element where it is displayed
        Parameters :
            source: path of the image file
            filepath: name of the image
    """
    
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
    """
        Utility function reduce the dimensions of the image, such as the maximum of the
        width and height is equal to max_dimension, and save it to the same path
        Parameters :
            source: path of the image file
            max_dimension: the maximum dimension of the image after reducing the size
    """

    im = Image.open(source)
    width, height = im.size

    ratio = max(width / max_dimension, height / max_dimension)
    if ratio > 1 :
        im = im.resize((int(width / ratio), int(height / ratio)))
        im.save(source)

def getMask(image_width, image_height, left, top, width, height) :
    """
        Utility function to get rectangular mask from json string of dash canvas 
        Parameters :
            image_width: width of the image
            image_height: height of the image
            left: offset from the left
            top: offset from the top
            width: width of the masked region
            height: height of the masked region
    """

    mask = np.ones((image_height, image_width))
    mask[top : top + height + 1, left : left + width + 1] = 0

    return mask

def readMask(mask_filepath) :
    """
        Utility function to read mask for a file
        Parameters :
            mask_filepath: the path of the mask
    """

    mask = np.array(Image.open(mask_filepath))
    if len(mask.shape) >= 3 :
        mask = 1 * (mask == np.min(mask))[:, :, 0]

    return mask

# Utility function to get mask from data of dash canvas
def maskFromData(string, data, mask_filepath, image_width, image_height, canvas_width, rect_fill) :
    """
        Parse the json object file and get the mask from it, then return it
        Parameters :
            string: data in string format 
            data: object that holds the canvas variables
            mask_filepath: file path to save the mask to
            image_width: width of the image
            image_height: height of the image
            canvas_width: width of the canvas
            rect_fill: boolean denoting whether or not to fill rectangle objects in mask
    """

    mask = np.zeros((image_height, image_width))

    # get rectangular mask objects and fill them in the mask, 
    # otherwise if parsed it's just the outline
    if 'fill' in rect_fill : 
        for object in data['objects'] :
            if object["type"] == "rect" :
                    
                left = int(object["left"] * image_width / canvas_width)
                top = int(object["top"] * image_width / canvas_width)
                width = int(object["width"] * image_width / canvas_width)
                height = int(object["height"] * image_width / canvas_width)

                rect_mask = getMask(image_width, image_height, left, top, width, height)
                mask[rect_mask == 0] = 1
    
    # parse the rest of the mask objects
    mask[parse_jsonstring(string, (image_height, image_width)) == 1] = 1

    plt.imsave(os.getcwd() + mask_filepath, mask, cmap=cm.gray)

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
    """
        Main function that handles the inpainting logic, and returning the resulting, impainted image.
        Parameters : 
            image: original image
            mask: binary image, 1's denote the masked region and 0's denote the rest of the image
            patch_size: patch size to use in the inpainting
            local_radius: local search radius to use in the inpainting
            data_significance: significance of the data term to use in the algorithm
            threshold: center similarity threshold to use in the algorithm
            live_update: whether or not to return intermediate results after each iteration
                (not implemented yet) 
            inpainted_filepath: the file path where to save the resulting image
            progress_filepath: the file path where to save the algorithm's progress (not used currently)
    """

    inpainter = Inpainter(patch_size, local_radius = local_radius, data_significance = data_significance, threshold = threshold)

    start_time = time.time()
    seconds_passed = 0
    inpainted = None
    # for inpainted, mask, progress in inpainter.inpaintWithSteps(image, mask) :

    #     # Update progress
    #     # with open(os.getcwd() + progress_filepath, 'w') as fp :
    #     #     fp.write(str(progress))

    #     if live_update :
    #         # Update image every 1s
    #         elapsed = int(time.time() - start_time)

    #         if elapsed > seconds_passed :
    #             inpainted[mask == 0] = 0
    #             # plt.imsave(os.getcwd() + inpainted_filepath, inpainted)
    #             seconds_passed = elapsed
    #             # print("elapsed : ", elapsed)

    inpainted = inpainter.inpaint(image, mask)
    plt.imsave(os.getcwd() + inpainted_filepath, inpainted)