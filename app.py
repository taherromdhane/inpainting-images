import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from PIL import Image, ImageOps
import json
import base64
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import uuid 
import time

from utils import getMask
from inpaint import inpaint
from layout import *

PREVIEW_HEIGHT = '500px'

app = dash.Dash(__name__, serve_locally=False, title='Inpainter')


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta content="width=device-width, initial-scale=1.0" name="viewport">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%} 
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


main_layout =   html.Div([
                    upload_layout
                ],
                id = 'main-div',
                className = 'main')

app.layout = html.Div([
    navbar_layout,
    header_layout,
    main_layout,
    footer_layout
    ])

# Utility function to display contents with uploaded image
def save_image(contents, filename):
    
    data = contents.encode("utf8").split(b";base64,")[1]

    with open(os.getcwd() + app.get_asset_url(filename), "wb") as fp:
        fp.write(base64.decodebytes(data))  

# Navbar callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

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

# Upload callback
@app.callback(
    [
        Output('upload-preview', 'children'), 
        Output('upload-image', 'style'), 
        Output('main-div', 'children'),        
        Output('upload-text', 'children')
    ],
    [
        Input('inpaint-button', 'n_clicks'),
        Input('upload-image', 'contents'),
        Input('change-upload', 'n_clicks')
    ], 
    [
        State('upload-image', 'filename'), 
        State('upload-image', 'last_modified')
    ]
)
def update_output(inpaint_clicks, image_content, change_clicks, name, date):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'inpaint-button' not in changed_id :
        if image_content is None :
            raise PreventUpdate
            
        else :
            return parse_contents(image_content, name), {'height': '100px'}, dash.no_update, 'Change Image'

    else :
        if image_content is None:
            raise PreventUpdate

        else :
            print(name)

            image_ID = str(uuid.uuid1())
            image_filename = 'source/' + image_ID + '.png'
            save_image(image_content, image_filename)
            
            inpaint_layout = getInpaintLayout(image_content, app.get_asset_url(image_filename))

            return dash.no_update, dash.no_update, inpaint_layout, dash.no_update
    
# Callbacks for inpainting parameters 
@app.callback(
    Output('annot-canvas', 'lineWidth'),
    [Input('brush-width-slider', 'value')], 
    prevent_initial_call=True
)
def update_canvas_linewidth(value):
    return value

@app.callback(
    Output('local-radius-output', 'children'),
    [Input('local-radius-slider', 'value')], 
    prevent_initial_call=True
)
def update_output(value):
    return '{}px'.format(value)

@app.callback(
    Output('data-term-output', 'children'),
    [Input('data-term-slider', 'value')], 
    prevent_initial_call=True
)
def update_output(value):
    return value

@app.callback(
    Output('center-similarity-output', 'children'),
    [Input('center-similarity-slider', 'value')], 
    prevent_initial_call=True
)
def update_output(value):
    return value


# Main callback for the inpainting
def mask_from_data(string, data, mask_filename, image_width, image_height, canvas_width, rect_fill) :
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

    plt.imsave(os.getcwd() + app.get_asset_url(mask_filename), mask, cmap=cm.gray)

@app.callback(
    Output('result-div', 'children'), 
    [Input('annot-canvas', 'json_data'),
    Input('annot-canvas', 'filename'),
    Input('upload-mask', 'contents'),
    Input('patch-size-slider', 'value'), 
    Input('local-radius-slider', 'value'),  
    Input('data-term-slider', 'value'), 
    Input('center-similarity-slider', 'value'),
    Input('fill-rect', 'value'),    
    Input('use-data-term', 'value'),    
    Input('use-center-threshold', 'value')],
    prevent_initial_call=True)
def inpaint_image(string, image_filename, mask_contents, patch_size, local_radius, data_significance, threshold, rect_fill, use_data, use_threshold):
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
        mask_from_data(string, data, mask_filename, image_width, image_height, CANVAS_WIDTH, rect_fill)
    
    mask = np.array(Image.open(os.getcwd() + app.get_asset_url(mask_filename)))
    mask = 1 * (mask == np.min(mask))[:, :, 0]
    print(mask.shape)

    # print(mask[0, 0, :])
    print(patch_size, local_radius, data_significance)

    if 'use' not in use_data :
        data_significance = 0

    if 'use' not in use_threshold :
        threshold = None

    start_time = time.time()
    inpainted = inpaint(image, mask, patch_size, local_radius = local_radius, data_significance = data_significance, threshold = 0.3)
    print("ran inpainting algorithm in {} : ".format(time.time() - start_time))

    inpainted_filename = 'results/' + image_ID + '.png'
    plt.imsave(os.getcwd() + app.get_asset_url(inpainted_filename), inpainted)

    return [
                html.H3('Result'),
                html.Hr(style = {'width' : '80%'}),
                html.Img(
                    id='image', 
                    width=CANVAS_WIDTH,
                    src = app.get_asset_url(inpainted_filename)
                )
            ]
    
    
app.css.append_css({
    'external_url': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css'
    ]
})

app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)