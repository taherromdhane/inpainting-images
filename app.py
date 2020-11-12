import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash_canvas import DashCanvas
import json
from dash_table import DataTable
from dash_canvas.utils import parse_jsonstring
import dash_core_components as dcc
from PIL import Image
import base64
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import uuid 
from utils import getMask
from inpaint import inpaint

app = dash.Dash(__name__, serve_locally=False)

filename = 'test_images/pavage.png'
canvas_width = 400
width = canvas_width

app.layout = html.Div([
    html.Div([
        html.H2('Draw on image and press Save to show annotations geometry'),
        html.Div([
            html.H6(children=['Brush width']),
            dcc.Slider(
                id='brush-width-slider',
                min=2,
                max=60,
                step=1,
                value=10
            ),
        ], className="three columns"),
        html.Div([
            html.H6(children=['Patch Size']),
            dcc.Slider(
                id='patch-size-slider',
                min=3,
                max=11,
                step=2,
                value=5,
                marks={3:'3', 5:'5', 7:'7', 9:'9', 11:'11'},
                included=False
            ),
        ], className="three columns"),
        html.Div([
            html.H6(children=['Local Radius']),
            dcc.Slider(
                id='local-radius-slider',
                min=50,
                max=500,
                step=5,
                value=50,
                marks={3:'3', 5:'5', 7:'7', 9:'9', 11:'11'},
                included=False
            ),
            html.Div(id='local-radius-output')
        ], className="two columns")
    ], className="row"),
    html.Div([
        html.Div([
            html.H3('Image to be filled'),
            DashCanvas(
                id='annot-canvas',
                lineWidth=10,
                filename=app.get_asset_url(filename),
                width=canvas_width,
                lineColor='rgba(255, 0, 0, 0.6)',
                tool="rectangle",
                hide_buttons=['line', 'zoom', 'pan', 'select'],
                goButtonTitle="Fill Mask"
                )], 
            className="six columns"),
        html.Div([
            html.H3('Result'),
            html.Img(id='image', width=canvas_width)
        ], className="six columns")
    ], className="row")
    ])

@app.callback(Output('annot-canvas', 'lineWidth'),
            [Input('brush-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

@app.callback(
    dash.dependencies.Output('local-radius-output', 'children'),
    [dash.dependencies.Input('local-radius-slider', 'value')])
def update_output(value):
    return '{}px'.format(value)

@app.callback(Output('image', 'src'), [Input('annot-canvas', 'json_data'), Input('patch-size-slider', 'value')])
def update_image(string, patch_size):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate

    print(data['objects'], file = sys.stderr)

    image_ID = str(uuid.uuid1())

    image_width = int(data['objects'][0]["width"])
    image_height = int(data['objects'][0]["height"])
    
    mask_filename = 'masks/' + image_ID + '.png'

    mask = np.zeros((image_height, image_width))

    if data['objects'][1]["type"] == "rect" :
            
        left = int(data['objects'][1]["left"] * image_width / canvas_width)
        top = int(data['objects'][1]["top"] * image_width / canvas_width)
        width = int(data['objects'][1]["width"] * image_width / canvas_width)
        height = int(data['objects'][1]["height"] * image_width / canvas_width)

        mask = getMask(image_width, image_height, left, top, width, height)
        # mask=Image.open(os.getcwd() + app.get_asset_url(mask_filename))
    # print(mask)
    else :
        mask =  1 - parse_jsonstring(string, (image_height, image_width))
    print(mask.shape)
    
    plt.imsave(os.getcwd() + app.get_asset_url(mask_filename), mask)

    image_filename = 'test_images/pavage.png'
    image = Image.open(os.getcwd() + app.get_asset_url(image_filename))

    # mask = cv2.imread(os.getcwd() + app.get_asset_url('mask2.jpg'),2)
    # ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    # print(mask)
    #plt.imsave(os.getcwd() + app.get_asset_url("mask2.jpg"), mask)


    inpainted = inpaint(image, mask, patch_size, local_radius=50)

    inpainted_filename = 'results/' + image_ID + '.png'

    # inpainted_array = Image.fromarray(inpainted)
    # Image.save(inpainted, "/assets/inpainted.jpeg"

    plt.imsave(os.getcwd() + app.get_asset_url(inpainted_filename), inpainted)

    return app.get_asset_url(inpainted_filename)
    
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)