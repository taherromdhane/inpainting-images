import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring



def getInpaintLayout(image_content, image_filename, CANVAS_WIDTH, CANVAS_HEIGHT) :
    canvas_layout = html.Div([
                        html.Div([
                            html.H3('Image to be filled'),
                            html.Hr(style = {'width' : '80%'}),
                            html.Div(
                                DashCanvas(
                                    id='annot-canvas',
                                    lineWidth=10,
                                    width=CANVAS_WIDTH,
                                    height=CANVAS_HEIGHT,
                                    lineColor='rgba(255, 0, 0, 0.6)',
                                    #image_content=image_content,
                                    filename=image_filename,
                                    tool="rectangle",
                                    hide_buttons=['line', 'pan', 'select'],
                                    goButtonTitle="Fill Mask",
                                    ),
                                className = 'col-md-10'
                            ),
                            dcc.Checklist(
                                id='fill-rect',
                                options=[
                                    {'label': 'Fill Rectangles', 'value': 'fill'},
                                ],
                                value=['fill']
                                ),
                            dcc.Checklist(
                                id='use-data-term',
                                options=[
                                    {'label': 'Use Data Term', 'value': 'use'},
                                ],
                                value=['use']
                                ),
                            dcc.Checklist(
                                id='use-center-threshold',
                                options=[
                                    {'label': 'Use Center Similarity Threshold', 'value': 'use'},
                                ],
                                value = []
                                ),
                            dcc.Checklist(
                                id='live-preview',
                                options=[
                                    {'label': 'Live Preview', 'value': 'show'},
                                ],
                                value = []
                                ),
                            html.Div([
                                html.Div(id='inpaint-alert-div'),
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
                                    multiple = False,
                                    className = 'upload-section'
                                )
                            ])
                        ], 
                        className = 'col-xl-6 canvas-section',
                        id = 'canvas_div'),
                        html.Div(
                            className = 'col-xl-6',
                            id = 'result-div',
                            style = {
                                'marginBottom' : '20px'
                        }),
                    ],
                    className = 'row')

    sliders_layout = html.Div([
                            html.Div([
                                html.H5(children=['Brush width']),
                                dcc.Slider(
                                    id='brush-width-slider',
                                    min=2,
                                    max=60,
                                    step=1,
                                    value=10
                                ),
                            ], 
                            className = "col-lg-2 sliders",
                            style = {'text-align':'center'}),
                            html.Div([
                                html.H5(children=['Patch Size']),
                                dcc.Slider(
                                    id='patch-size-slider',
                                    min=3,
                                    max=13,
                                    step=2,
                                    value=9,
                                    marks={3:'3', 5:'5', 7:'7', 9:'9', 11:'11', 13:'13'},
                                    included=False
                                ),
                            ], 
                            className = "col-lg-3 sliders",
                            style = {'text-align':'center'}),
                            html.Div([
                                html.H5(children=[
                                    'Local Radius : ',
                                    html.Span('100px', id='local-radius-output')
                                ]),
                                dcc.Slider(
                                    id='local-radius-slider',
                                    min=50,
                                    max=500,
                                    step=5,
                                    value=100,
                                    included=False
                                )
                            ], 
                            className = "col-lg-3 sliders"),
                            html.Div([
                                html.H5(children=[
                                    'Data Term significance : ',                                    
                                    html.Span('1', id='data-term-output')
                                ]),
                                dcc.Slider(
                                    id='data-term-slider',
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=1,
                                    marks={0:'0', 1:'1'}
                                ),
                            ],
                            className = "col-lg-2 sliders",
                            style = {'text-align' : 'center'}),
                            html.Div([
                                html.H5(children=[
                                    'Center Similarity Threshold : ',                                    
                                    html.Span('0.7', id='center-similarity-output')
                                ]),
                                dcc.Slider(
                                    id='center-similarity-slider',
                                    min=0.1,
                                    max=1.8,
                                    step=0.01,
                                    value=0.7,
                                    marks={0.1:'0.1', 1.8:'1.8'}
                                ),
                            ],
                            className = "col-lg-2 sliders",
                            style = {'text-align' : 'center'})
                        ], 
                        className = "row",
                        style = {'text-align' : 'center', 'width' : '100%', 'marginTop' : '20px', 'marginBottom' : '20px'})

    inpaint_layout = html.Div([
                        html.H1(
                            'Draw Mask and Then Run The Inpainter',
                            style = {
                                'text-align':'center',
                                'margin' : '40px 0px'
                        }),
                        sliders_layout,
                        canvas_layout
                    ], 
                    className = 'col')
                    
    return inpaint_layout