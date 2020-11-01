import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash_canvas import DashCanvas
import json
from dash_table import DataTable
from PIL import Image
import base64
import os
import sys
import matplotlib.pyplot as plt
import cv2
from utils import getMask
from inpaint import inpaint

app = dash.Dash(__name__)

filename = 'simp.png'
canvas_width = 600
width = canvas_width

app.layout = html.Div([
    html.H6('Draw on image and press Save to show annotations geometry'),
    DashCanvas(id='annot-canvas',
               lineWidth=5,
               filename=app.get_asset_url(filename),
               width=canvas_width,
               ),
    html.Img(id='image', width=canvas_width)])

@app.callback(Output('image', 'src'), [Input('annot-canvas', 'json_data')])
def update_image(string):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate

    print(data['objects'], file = sys.stderr)

    image_width = int(data['objects'][0]["width"])
    image_height = int(data['objects'][0]["height"])

    left = int(data['objects'][1]["left"] * image_width / canvas_width)
    top = int(data['objects'][1]["top"] * image_width / canvas_width)
    width = int(data['objects'][1]["width"] * image_width / canvas_width)
    height = int(data['objects'][1]["height"] * image_width / canvas_width)

    mask = getMask(image_width, image_height, left, top, width, height)
    # mask_filename='mask1.jpg'
    # mask=Image.open(os.getcwd() + app.get_asset_url(mask_filename))
    # print(mask)
    plt.imsave(os.getcwd() + app.get_asset_url("mask.jpg"), mask)

    image_filename = 'simp.png'
    image = Image.open(os.getcwd() + app.get_asset_url(image_filename))

    # mask = cv2.imread(os.getcwd() + app.get_asset_url('mask2.jpg'),2)
    # ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    # print(mask)
    #plt.imsave(os.getcwd() + app.get_asset_url("mask2.jpg"), mask)


    inpainted = inpaint(image, mask, local_radius=50)

    inpainted_filename = "tuto.PNG"

    # inpainted_array = Image.fromarray(inpainted)
    # Image.save(inpainted, "/assets/inpainted.jpeg"

    plt.imsave(os.getcwd() + app.get_asset_url(inpainted_filename), inpainted)

    return app.get_asset_url(inpainted_filename)


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)