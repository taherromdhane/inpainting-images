import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash_canvas import DashCanvas
import json
from dash_table import DataTable
from dash_canvas.utils import parse_jsonstring
import dash_core_components as dcc
from PIL import Image, ImageOps
import base64
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import uuid 
from utils import getMask
from inpaint import inpaint

app = dash.Dash(__name__, serve_locally=False)
canvas_width = 500
width = canvas_width

app.layout = html.Div([
    html.H2(
        'Draw on image and press Save to show annotations geometry',
        style = {
            'text-align':'center',
            'margin' : '40px'
        }),
    html.Div([
        dcc.Upload(
            id = 'upload-image',
            children = html.Div([
                'Drag and Drop or ',
                html.A('Select Images')
            ]),
            style = {
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'position': 'center'
            },
            accept = '.png, .jpg, .jpeg',
            # don't allow multiple files to be uploaded
            multiple = False
        )
    ]),
    html.Div([
        html.Div([
            html.H6(children=['Brush width']),
            dcc.Slider(
                id='brush-width-slider',
                min=2,
                max=60,
                step=1,
                value=10
            ),
        ], 
        className = "three columns",
        style = {'text-align':'center'}),
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
        ], 
        className = "three columns",
        style = {'text-align':'center'}),
        html.Div([
            html.H6(children=['Local Radius']),
            dcc.Slider(
                id='local-radius-slider',
                min=50,
                max=500,
                step=5,
                value=100,
                marks={3:'3', 5:'5', 7:'7', 9:'9', 11:'11'},
                included=False
            ),
            html.Div(id='local-radius-output')
        ], 
        className = "two columns"),
        html.Div([
            html.H6(children=['Data Term significance']),
            dcc.Slider(
                id='data-term-slider',
                min=0,
                max=1,
                step=0.01,
                value=1,
                marks={0:'0', 1:'1'}
            ),
        ],
        className = "two columns",
        style = {'text-align' : 'center', 'display' : 'none'})
    ], 
    className = "row",
    style = {'text-align':'center'}),
    html.Div([
        html.Div([
            html.H3('Image to be filled'),
            DashCanvas(
                id='annot-canvas',
                lineWidth=10,
                width=canvas_width,
                lineColor='rgba(255, 0, 0, 0.6)',
                tool="rectangle",
                hide_buttons=['line', 'zoom', 'pan', 'select'],
                goButtonTitle="Fill Mask"
                ),
            dcc.Checklist(
                id='fill-rect',
                options=[
                    {'label': 'Fill Rectangles', 'value': 'fill'},
                ],
                value=['fill']
                ),
            html.Div([
                dcc.Upload(
                    id = 'upload-mask',
                    children = html.Div([
                        'Or ',
                        html.A('Upload a Mask')
                    ]),
                    style = {
                        'width': '80%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'position': 'center'
                    },
                    accept = '.png, .jpg, .jpeg',
                    # don't allow multiple files to be uploaded
                    multiple = False
                )
            ])
            ], 
            className = "six columns",
            id = "canvas_div"),
        html.Div([
            html.H3('Result'),
            html.Img(
                id='image', 
                width=canvas_width
            )
        ], 
        className = "six columns")
    ], 
    className = "row")
    ],
    style = {'margin' : '20px'})

# Utility function to display contents with uploaded image
def save_image(contents, filename):
    
    data = contents.encode("utf8").split(b";base64,")[1]

    with open(os.getcwd() + app.get_asset_url(filename), "wb") as fp:
        fp.write(base64.decodebytes(data))  

# Upload callback
@app.callback(Output('canvas_div', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(image_content, name, date):
    
    if image_content is not None:
        
        print(name)

        image_ID = str(uuid.uuid1())
        image_filename = 'source/' + image_ID + '.png'
        save_image(image_content, image_filename)

        return  html.Div([
                    html.H3('Image to be filled'),
                    DashCanvas(
                        id='annot-canvas',
                        lineWidth=10,
                        width=canvas_width,
                        lineColor='rgba(255, 0, 0, 0.6)',
                        filename=app.get_asset_url(image_filename),
                        tool="rectangle",
                        hide_buttons=['line', 'zoom', 'pan', 'select'],
                        goButtonTitle="Fill Mask"
                        ),
                    dcc.Checklist(  
                        id='fill-rect',
                        options=[
                            {'label': 'Fill Rectangles', 'value': 'fill'},
                        ],
                        value=['fill']),
                    html.Div([
                        dcc.Upload(
                            id = 'upload-mask',
                            children = html.Div([
                                'Or ',
                                html.A('Upload a Mask')
                            ]),
                            style = {
                                'width': '80%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'position': 'center'
                            },
                            accept = '.png, .jpg, .jpeg',
                            # don't allow multiple files to be uploaded
                            multiple = False
                        )
                    ])
                    ], 
                    className = "six columns")
    else:
        raise PreventUpdate
    
# Callbacks for inpainting parameters 
@app.callback(Output('annot-canvas', 'lineWidth'),
            [Input('brush-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

@app.callback(
    Output('local-radius-output', 'children'),
    [Input('local-radius-slider', 'value')])
def update_output(value):
    return '{}px'.format(value)

# Main callback for the inpainting
def mask_from_data(string, data, mask_filename, image_width, image_height, canvas_width, rect_fill) :
    mask = np.ones((image_height, image_width))

    if 'fill' in rect_fill : 
        for object in data['objects'] :
            if object["type"] == "rect" :
                    
                left = int(object["left"] * image_width / canvas_width)
                top = int(object["top"] * image_width / canvas_width)
                width = int(object["width"] * image_width / canvas_width)
                height = int(object["height"] * image_width / canvas_width)

                rect_mask = getMask(image_width, image_height, left, top, width, height)
                mask[rect_mask == 0] = 0
    
    mask[parse_jsonstring(string, (image_height, image_width)) == 1] = 0 

    plt.imsave(os.getcwd() + app.get_asset_url(mask_filename), mask, cmap=cm.gray)

@app.callback(
    Output('image', 'src'), 
    [Input('annot-canvas', 'json_data'),
    Input('annot-canvas', 'filename'),
    Input('upload-mask', 'contents'),
    Input('patch-size-slider', 'value'), 
    Input('local-radius-slider', 'value'),  
    Input('data-term-slider', 'value'), 
    Input('fill-rect', 'value')])
def inpaint_image(string, image_filename, mask_contents, patch_size, local_radius, data_significance, rect_fill):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate
    
    # print(data)

    image_ID = image_filename.split('/')[3].split('.')[0]
    image = np.array(Image.open(os.getcwd() + image_filename))

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    mask_filename = 'masks/' + image_ID + '.png'
    print(mask_filename)

    if mask_contents is not None :
        save_image(mask_contents, mask_filename)
    else : 
        mask_from_data(string, data, mask_filename, image_width, image_height, canvas_width, rect_fill)
    
    mask = np.array(Image.open(os.getcwd() + app.get_asset_url(mask_filename)))
    mask = 1 * (mask == np.max(mask))[:, :, 0]
    print(mask.shape)

    # print(mask[0, 0, :])

    inpainted = inpaint(image, mask, patch_size, local_radius = local_radius, data_significance = data_significance)
    inpainted_filename = 'results/' + image_ID + '.png'
    plt.imsave(os.getcwd() + app.get_asset_url(inpainted_filename), inpainted)

    return app.get_asset_url(inpainted_filename)
    
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)