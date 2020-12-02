import numpy as np
import dash_html_components as html
from dash_canvas.utils import parse_jsonstring
import base64

# Utility function to save contents to memory
def save_image(contents, filepath):
    
    data = contents.encode("utf8").split(b";base64,")[1]

    with open(os.getcwd() + filepath, "wb") as fp:
        fp.write(base64.decodebytes(data))  


# Utility function to show preview of uploaded file
def parse_contents(contents, filename):
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


# Utility function to get rectangular mask from dash canvas
def getMask(image_width, image_height, left, top, width, height) :

    mask = np.ones((image_height, image_width))

    mask[top : top + height + 1, left : left + width + 1] = 0

    return mask


# Utility function to get mask from data of dash canvas
def mask_from_data(string, data, mask_filepath, image_width, image_height, canvas_width, rect_fill) :
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

def inpainting_logic(image, mask, patch_size, local_radius, data_significance, threshold, live_update, inpainted_filepath, progress_filepath)

    inpainter = Inpainter(patch_size, local_radius = local_radius, data_significance = data_significance, threshold = threshold)

    start_time = time.time()
    seconds_passed = 0
    for inpainted, progress in inpainter.inpaint_with_progress(image, mask) :

        # Update progress
        with open(progress_filepath, 'w') as fp :
            fp.write(progress)

        if live_update :

            # Update image every 1s
            elapsed = int(time.time() - start_time)
            if elapsed > seconds_passed :
                plt.imsave(os.getcwd() + inpainted_filepath, inpainted)
                seconds_passed = elapsed