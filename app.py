import flask
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
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import uuid 
import time

from gevent.pywsgi import WSGIServer

from utils import getMask
from inpaint import Inpainter
from layout import *

from rq import Queue
from rq.job import Job
from worker import conn

PREVIEW_HEIGHT = '500px'

server = flask.Flask(__name__) # define flask app.server
app = dash.Dash(__name__, title='Inpainter', eager_loading=True, server=server)

q = Queue(connection=conn)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta content="width=device-width, initial-scale=1.0" name="viewport">
        <title>{%title%}</title>
        {%favicon%}
        <link rel="stylesheet" href="https://codepen.io/chriddyp/pen/bWLwgP.css" crossorigin="anonymous">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
        <link href="https://maxcdn.bootstrapcdn.com/font-awesome/latest/css/font-awesome.min.css" rel="stylesheet">
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
        dcc.Store(id='session', storage_type='session'),
        navbar_layout,
        header_layout,
        main_layout,
        footer_layout
    ])



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

# Upload callbacks
@app.callback(
    [
        Output('upload-preview', 'children'), 
        Output('upload-image', 'style'), 
        Output('main-div', 'children'),        
        Output('upload-text', 'children'),
        Output('session', 'data')
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
            return parse_contents(image_content, name), {'height': '100px'}, dash.no_update, 'Change Image', dash.no_update

    else :
        if image_content is None:
            raise PreventUpdate

        else :
            print(name)

            image_ID = str(uuid.uuid1())
            image_filename = 'source/' + image_ID + '.png'
            save_image(image_content, app.get_asset_url(image_filename))
            
            inpaint_layout = getInpaintLayout(image_content, app.get_asset_url(image_filename))

            return dash.no_update, dash.no_update, inpaint_layout, dash.no_update, {'image_ID': image_ID}
    
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


# Main callback for inpainting
@app.callback(
    Output('result-div', 'children'), 
    [
        Input('annot-canvas', 'json_data'),
        Input('session', 'modified_timestamp'),
        Input('upload-mask', 'contents'),
        Input('patch-size-slider', 'value'), 
        Input('local-radius-slider', 'value'),  
        Input('data-term-slider', 'value'), 
        Input('center-similarity-slider', 'value'),
        Input('fill-rect', 'value'),    
        Input('use-data-term', 'value'),    
        Input('use-center-threshold', 'value')
    ],
    State('session', 'data'),
    prevent_initial_call=True)
def inpaint_image(string, ts, mask_contents, patch_size, local_radius, data_significance, \
                                        threshold, rect_fill, use_data, use_threshold, session_data):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate

    if ts is None:
            raise PreventUpdate

    image_ID = session_data.get('image_ID', '')
    image_filename = 'source/' + image_ID + '.png'
    image = np.array(Image.open(os.getcwd() + app.get_asset_url(image_filename)))

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    mask_filename = 'masks/' + image_ID + '.png'
    print(mask_filename)

    if mask_contents is not None :
        save_image(mask_contents, app.get_asset_url(mask_filename))
    else : 
        mask_from_data(string, data, app.get_asset_url(mask_filename), image_width, image_height, CANVAS_WIDTH, rect_fill)
    
    mask = np.array(Image.open(os.getcwd() + app.get_asset_url(mask_filename)))
    if len(mask.shape) >= 3 :
        mask = 1 * (mask == np.min(mask))[:, :, 0]

    # print(mask[0, 0, :])

    if 'use' not in use_data :
        data_significance = 0

    if 'use' not in use_threshold :
        threshold = None

    start_time = time.time()


    inpainted_filename = 'results/' + image_ID + '.png'
    progress_filename =  'progress/' + image_ID + '.txt'

    predict_job = q.enqueue(inpainting_logic, image, mask, patch_size, local_radius, data_significance, threshold, app.get_asset_url(inpainted_filename))

    return [
                html.H3('Result'),
                html.Hr(style = {'width' : '80%'}),
                html.Div(
                    [
                        dcc.Interval(id="progress-interval", n_intervals=0, interval=100),
                        dbc.Progress(id="progress", striped=True, animated=True, value=0),
                    ]
                ),
                html.Div(
                    [
                        dcc.Interval(id="result-interval", n_intervals=0, interval=1000),
                        html.Img(
                            id='result-image', 
                            width=CANVAS_WIDTH,
                            src = app.get_asset_url(inpainted_filename)
                        )
                    ]
                )   
            ] 


@app.callback(
    [Output("progress", "value"), Output("progress", "children")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    progress = min(n % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 5 else ""



    inpainted_filename = 'results/' + image_ID + '.png'
    progress_filename =  'progress/' + image_ID + '.txt'
    
    print("ran inpainting algorithm in {} : ".format(time.time() - start_time))

    

    
    
    
app.css.append_css({
    'external_url': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css'
    ]
})

app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    # app.run_server(debug=True, dev_tools_hot_reload = False)

    http_server = WSGIServer(('0.0.0.0', int(os.environ.get("PORT", 5000))), server)
    print(int(os.environ.get("PORT", 5000)), file=sys.stderr)
    http_server.serve_forever()