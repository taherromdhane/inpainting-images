import flask
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc


from PIL import Image
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import uuid 
import time
import shutil
from datetime import datetime


from gevent.pywsgi import WSGIServer

from utils import *
from inpaint import Inpainter
from layout import *

PREVIEW_HEIGHT = '500px'
TEST_DIR = 'test_images'
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 800
MAX_DIMENSION = 300

# Initialize app
server = flask.Flask(__name__) # define flask app.server
app = dash.Dash(__name__, title='Inpainter', eager_loading=True, server=server)

# Initialize Redis Queue
# from rq import Queue
# from rq.job import Job
# from worker import conn

# q = Queue(connection=conn)

# Modify app layout to include bootstrap and custom css
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

# Define the app and main layouts
main_layout =   html.Div([
                    getUploadLayout(app.get_asset_url(TEST_DIR))
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
## Sample images selector callback
@app.callback(
    Output('sample-image-chosen', 'children'),
    Input({'type': 'sample-img', 'source': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def set_chosen_img(n_clicks) :
    print(dash.callback_context.triggered, file = sys.stderr)
    change_id = dash.callback_context.triggered[0]['prop_id'].split('.')[:-1]
    json_source = ".".join(dash.callback_context.triggered[0]['prop_id'].split('.')[:-1])
    if change_id:
        return [json.loads(json_source)["source"], time.time()]
    else :
        return ["", time.time()]

## Upload time callback
@app.callback(
    Output('upload-time-div', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def set_upload_time(content) :
    return time.time()

## Upload logic callback
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
        # Input('change-upload', 'n_clicks'),
        Input('sample-image-chosen', 'children'),
        Input('upload-time-div', 'children')
    ], 
    [
        State('upload-image', 'filename')
    ],
    prevent_initial_call=True
)
def update_output(inpaint_clicks, image_content, sample_img_data, date_upload, name):
# def update_output(inpaint_clicks, image_content, change_clicks, n_clicks, name, date, source):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)

    print(sample_img_data)
    if sample_img_data :
        source, date_sample = sample_img_data[0], sample_img_data[1]
    else :
        source, date_sample = None, None

    if 'inpaint-button' not in changed_id :
        if 'upload-image.contents' in changed_id :
            print(date_upload)
            return parseContents(image_content, name), {'height': '100px'}, dash.no_update, 'Change Image', dash.no_update
            
        elif 'sample-image-chosen.children' in changed_id :
            print(source)
            return parseContentsDir(app.get_asset_url(TEST_DIR + '/' + source), name), {'height': '100px'}, dash.no_update, 'Change Image', dash.no_update
        
        else :
            raise PreventUpdate

    else :
        if image_content is not None or source is not None:
            print(date_upload, date_sample)

            image_ID = str(uuid.uuid1())
            image_filename = 'source/' + image_ID + '.png'

            if date_upload is not None and (date_sample is None or date_upload > date_sample) :
                saveImage(image_content, app.get_asset_url(image_filename))
                reduceImageSize(app.get_asset_url(image_filename)[1:], max_dimension)
                
                inpaint_layout = getInpaintLayout(image_content, app.get_asset_url(image_filename), CANVAS_WIDTH, CANVAS_HEIGHT)

            else :
                shutil.copy(app.get_asset_url(TEST_DIR)[1:] + "/" + source, app.get_asset_url(image_filename)[1:])
                reduceImageSize(app.get_asset_url(image_filename)[1:], MAX_DIMENSION)

                inpaint_layout = getInpaintLayout(None, app.get_asset_url(image_filename), CANVAS_WIDTH, CANVAS_HEIGHT)

            return dash.no_update, dash.no_update, inpaint_layout, dash.no_update, {'image_ID': image_ID}

        else :
            raise PreventUpdate
    
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
        Input('use-center-threshold', 'value'),
        Input('live-preview', 'value')
    ],
    State('session', 'data'),
    prevent_initial_call=True)
def inpaint_image(string, ts, mask_contents, patch_size, local_radius, data_significance, \
                                        threshold, rect_fill, use_data, use_threshold, show_live_preview, session_data):
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate

    image_ID = session_data.get('image_ID', '')
    image_filename = 'source/' + image_ID + '.png'
    image = np.array(Image.open(os.getcwd() + app.get_asset_url(image_filename)))

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    mask_filename = 'masks/' + image_ID + '.png'
    print(mask_filename)

    if mask_contents is not None :
        saveImage(mask_contents, app.get_asset_url(mask_filename))
    else : 
        maskFromData(string, data, app.get_asset_url(mask_filename), image_width, image_height, CANVAS_WIDTH, rect_fill)
    
    mask = readMask((os.getcwd() + app.get_asset_url(mask_filename)))
    
    if 'use' not in use_data :
        data_significance = 0

    if 'use' not in use_threshold :
        threshold = None

    live_preview = False
    if 'show' in show_live_preview :
        live_preview = True

    start_time = time.time()

    inpainted_filename = 'results/' + image_ID + '.png'
    progress_filename =  'progress/' + image_ID + '.txt'

    inpainted = np.copy(image)
    print(time.strftime('%H:%M:%S', time.gmtime(time.time())))
    inpainted[mask == 0] = 0

    plt.imsave(os.getcwd() + app.get_asset_url(inpainted_filename), inpainted)

    with open(os.getcwd() + app.get_asset_url(progress_filename), 'w') as fp :
        fp.write("0")

    # predict_job = q.enqueue(
    #                     inpaintingLogic, 
    #                     image, 
    #                     mask, 
    #                     patch_size, 
    #                     local_radius, 
    #                     data_significance,
    #                     threshold, 
    #                     live_preview, 
    #                     app.get_asset_url(inpainted_filename), 
    #                     app.get_asset_url(progress_filename)
    #                 )
    
    inpaintingLogic(
        image,
        mask,
        patch_size,
        local_radius,
        data_significance,
        threshold,
        live_preview,
        app.get_asset_url(inpainted_filename),
        app.get_asset_url(progress_filename)
    )

    return [
                html.H3('Result'),
                html.Hr(style = {'width' : '80%'}),
                html.Div(
                    [
                        dcc.Interval(id="progress-interval", n_intervals=0, interval=100),
                        dbc.Progress(
                            id = "progress", 
                            striped = True, 
                            animated = True, 
                            value = 0,
                            style = {
                                "height": "30px",
                                "margin" : "10px"
                            }
                        ),
                    ],
                    id = "progress-interval-div"
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Interval(id="result-interval", n_intervals=0, interval=1000),
                            id = "result-interval-div"
                        ),
                        html.Div(
                            html.Img(
                                # id='result-image', 
                                width=CANVAS_WIDTH,
                                src = app.get_asset_url(inpainted_filename)
                            ),
                            id = 'result-image-div'
                        )
                    ]
                )   
            ] 

# Callback for updating the progress bar
@app.callback(
    [
        Output("progress", "value"), 
        Output("progress", "children"),
        Output("progress-interval-div", "children")
    ],
    [
        Input("progress-interval", "n_intervals")
    ],
    State('session', 'data'),
    prevent_initial_call=True
)
def update_progress(n_intervals, session_data):

    image_ID = session_data.get('image_ID', '')

    progress_filename =  'progress/' + image_ID + '.txt'
    with open(os.getcwd() + app.get_asset_url(progress_filename), 'r') as fp :
        progress = int(float(fp.read()))

    # progress = min(progress % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    if progress < 100 :
        return  progress, f"{progress} %" if progress >= 5 else "", dash.no_update

    else :
        return  progress, f"{progress} %" if progress >= 5 else "", None

    # print("ran inpainting algorithm in {} : ".format(time.time() - start_time))

# Callback for updating the live preview and showing the result
@app.callback(
    [ 
        Output("result-image-div", "children"),
        Output("result-interval-div", "children")
    ],
    [
        Input("result-interval", "n_intervals"),
        Input('live-preview', 'value')
    ],
    State('session', 'data'),
    prevent_initial_call=True
)

def update_preview(n_intervals, show_live_preview, session_data) :

    image_ID = session_data.get('image_ID', '')
    
    progress_filename =  'progress/' + image_ID + '.txt'
    with open(os.getcwd() + app.get_asset_url(progress_filename), 'r') as fp :
        progress = int(float(fp.read()))

    if progress <= 100 and 'show' not in show_live_preview :
        raise PreventUpdate
    
    inpainted_filename = 'results/' + image_ID + '.png'

    if progress == 100 :
        return  html.Img(
                    # id='result-image', 
                    width = CANVAS_WIDTH,
                    src = app.get_asset_url(inpainted_filename)
                ), None
    else :
        return  html.Img(
                    # id='result-image', 
                    width = CANVAS_WIDTH,
                    src = app.get_asset_url(inpainted_filename)
                ), dash.no_update


app.css.append_css({
    'external_url': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css'
    ]
})

app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False)

    # http_server = WSGIServer(('0.0.0.0', int(os.environ.get("PORT", 5000))), server)
    # print(int(os.environ.get("PORT", 5000)), file=sys.stderr)
    # http_server.serve_forever()