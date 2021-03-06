import flask
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import os
import sys
import matplotlib.pyplot as plt
import uuid 
import time
import shutil


from gevent.pywsgi import WSGIServer

from utils import *
from inpaint import Inpainter
from layout import *

PREVIEW_HEIGHT = '500px'
TEST_DIR = 'test_images'
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 800
MAX_DIMENSION = 200

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


# Navbar callback for toggling
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """
        Callback method to toggle the navbar
        Parameters :
            n: number of clicks on the navbar toggler
            is_open: state of the navbar, whether open or closed
    """
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
    """
        Callback method that stores the selected sample image in the 
        'sample-image-chosen' element for use later in preview and upload
        Parameters :
            n_clicks: clicks on any of the 'sample-img' elements (note the ALL)
    """

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
def set_upload_time(contents) :
    """
        Callback method that stores the time of the last upload
        Parameters :
            content: contents of the upload element
    """
    return time.time()

## Upload logic callback
@app.callback(
    [
        Output('upload-preview', 'children'), 
        Output('upload-image', 'style'), 
        Output('main-div', 'children'),        
        Output('upload-text', 'children'),
        Output('upload-alert-div', 'children'),
        Output('session', 'data')
    ],
    [
        Input('inpaint-button', 'n_clicks'),
        Input('upload-image', 'contents'),
        Input('sample-image-chosen', 'children'),
        Input('upload-time-div', 'children')
    ], 
    [
        State('upload-image', 'filename')
    ],
    prevent_initial_call=True
)
def update_output(inpaint_clicks, image_content, sample_img_data, date_upload, name):
    """
        Callback method that handles the update of the preview, whether from the 
        sample images or the upload element, and also handle the inpaint button
        logic
        Parameters :
            inpaint_clicks: clicks on the inpaint button
            image_content: contents of the upload element
            sample_img_data: last selected sample image and its source path
            date_upload: the date of the last image upload by user
            name: name of the uploaded image file
    """

    # Get the element(s) that fired the callback 
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # Initialize variables depending on whether a sample image was chosen or not
    if sample_img_data :
        source, date_sample = sample_img_data[0], sample_img_data[1]
    else :
        source, date_sample = None, None

    # If the trigger didn't come from the "inpaint" button
    if 'inpaint-button' not in changed_id :
        if 'upload-image.contents' in changed_id :
            # Get image contents from upload and show it in preview
            return parseContents(image_content, name), {'height': '100px'}, dash.no_update, 'Change Image', dash.no_update, dash.no_update
            
        elif 'sample-image-chosen.children' in changed_id :
            # Show the selected sample image in preview
            return parseContentsDir(source, name), {'height': '100px'}, dash.no_update, 'Change Image', dash.no_update, dash.no_update
        
        else :
            raise PreventUpdate
    
    # Handle the logic of the "inpaint" button
    else :
        if image_content is not None or source is not None:
            # If there is at least a sample image selected or an uploaded image
            # Get the latest of the sample image selected and the uploaded image
            # and save it, reduce its size, then generate the inpaint layout using it

            image_ID = str(uuid.uuid1())
            image_filename = 'source/' + image_ID + '.png'

            if date_upload is not None and (date_sample is None or date_upload > date_sample) :
                # If the uploaded image is the latest
                saveImage(image_content, app.get_asset_url(image_filename))
                reduceImageSize(app.get_asset_url(image_filename)[1:], MAX_DIMENSION)
                
                inpaint_layout = getInpaintLayout(image_content, app.get_asset_url(image_filename), CANVAS_WIDTH, CANVAS_HEIGHT)

            else :
                # If the sample image was selected last
                shutil.copy(source, app.get_asset_url(image_filename)[1:])
                reduceImageSize(app.get_asset_url(image_filename)[1:], MAX_DIMENSION)

                inpaint_layout = getInpaintLayout(None, app.get_asset_url(image_filename), CANVAS_WIDTH, CANVAS_HEIGHT)

            return dash.no_update, dash.no_update, inpaint_layout, dash.no_update, dash.no_update, {'image_ID': image_ID}

        else :
            # If no image was selected or uploaded
            alert = dbc.Alert("Please upload or choose image!", color="danger", duration=2000)
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, alert, dash.no_update
    
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
    [
        Output('inpaint-alert-div', 'children'),
        Output('result-div', 'children')
    ], 
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
    """
        Callback method that handles the update of the preview, whether from the 
        sample images or the upload element, and also handle the inpaint button
        logic
        Parameters :
            inpaint_clicks: clicks on the inpaint button
            image_content: contents of the upload element
            sample_img_data: last selected sample image and its source path
            date_upload: the date of the last image upload by user
            name: name of the uploaded image file
    """

    # Get data that holds canvas objects, including masks on the image
    if string:
        data = json.loads(string)
    else:
        raise PreventUpdate

    # Load image to run the algorithm on
    image_ID = session_data.get('image_ID', '')
    image_filename = 'source/' + image_ID + '.png'
    image = np.array(Image.open(os.getcwd() + app.get_asset_url(image_filename)))

    image_width = image.shape[1]
    image_height = image.shape[0]
    
    # Parse the mask from Canvas and save it to file
    mask_filename = 'masks/' + image_ID + '.png'
    print(mask_filename)

    if mask_contents is not None :
        saveImage(mask_contents, app.get_asset_url(mask_filename))
    else : 
        maskFromData(string, data, app.get_asset_url(mask_filename), image_width, image_height, CANVAS_WIDTH, rect_fill)
    
    mask = readMask((os.getcwd() + app.get_asset_url(mask_filename)))
    
    if mask.shape != image.shape[:2] :
        return dbc.Alert("Mask is not the same shape as image!", color="danger", duration=2000), dash.no_update

    # Give default values to variables
    if 'use' not in use_data :
        data_significance = 0

    if 'use' not in use_threshold :
        threshold = None

    live_preview = False
    if 'show' in show_live_preview :
        live_preview = True

    # Initialize result image
    inpainted_filename = 'results/' + image_ID + '.png'
    inpainted = np.copy(image)
    inpainted[mask == 0] = 0

    # Run the inpainting algorithm    
    start_time = time.time()

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

    print("ran inpainting algorithm in {} : ".format(time.time() - start_time))

    # return the resulting div, with the result image
    return dash.no_update, \
            [
                html.H3('Result'),
                html.Hr(style = {'width' : '80%'}),
                html.Div(
                    [
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

# Add external css for bootstrap
app.css.append_css({
    'external_url': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css'
    ]
})

app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    # To run server in local environment
    app.run_server(debug=True, dev_tools_hot_reload = False)

    # To run server when deployed
    # http_server = WSGIServer(('0.0.0.0', int(os.environ.get("PORT", 5000))), server)
    # print(int(os.environ.get("PORT", 5000)), file=sys.stderr)
    # http_server.serve_forever()