import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash_canvas import DashCanvas

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# navbar_layout = dbc.Navbar(
#                     children=[
#                         html.A(
#                             # Use row and col to control vertical alignment of logo / brand
#                             dbc.Row(
#                                 [
#                                     dbc.Col(html.Img(src=LOGO, height="30px")),
#                                     dbc.Col(dbc.NavbarBrand("Inpainter", className="ml-2")),
#                                 ],
#                                 align="center",
#                                 no_gutters=True,
#                             ),
#                             # href="https://plot.ly",
#                         ),
#                         dbc.NavItem(dbc.NavLink("Page 1", href="#")),
#                         dbc.DropdownMenu(
#                             children=[
#                                 dbc.DropdownMenuItem("More pages", header=True),
#                                 dbc.DropdownMenuItem("Page 2", href="#"),
#                                 dbc.DropdownMenuItem("Page 3", href="#"),
#                             ],
#                             nav=True,
#                             in_navbar=True,
#                             label="More",
#                         ),
#                         dbc.NavbarToggler(id="navbar-toggler")
#                     ],
#                     className = "navbar fixed-top",
#                     color="black",
#                     dark=True,
#                 )


navbar_layout = dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(
                            dbc.NavLink(
                                [html.I(className="fa fa-github"), "View Source Code"], 
                                href="https://github.com/taherromdhane/inpainting-images"
                                )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                [html.I(className="fa fa-file"), "View Report"], 
                                href="https://github.com/taherromdhane/inpainting-images"
                                )
                        ),
                        dbc.DropdownMenu(
                            children=[
                                dbc.DropdownMenuItem("More Links", header=True),
                                dbc.DropdownMenuItem("Github", href="https://github.com/taherromdhane"),
                                dbc.DropdownMenuItem("LinkedIn", href="https://www.linkedin.com/in/taher-romdhane/"),
                            ],
                            nav=True,
                            in_navbar=True,
                            label="More",
                        ),
                    ],
                    style = {
                        'fontWeight' : 'bold',
                        'height' : '50px',
                        'paddingRight' : '3rem',
                        'paddingLeft' : '3rem',
                    },
                    brand="The Inpainter",
                    brand_href="#",
                    color="black",
                    fluid = True,
                    sticky = 'top',
                    dark=True,
                )


header_layout = html.Div(
                    html.Div(
                        html.Div(
                            html.Div([
                                    html.H1(html.Strong("THE INPAINTER")),
                                    html.H3("Remove Anythin From Your Images !")
                                ],
                                className = "brand"
                            ),
                            className = "col-xl-12 ml-auto mr-auto"
                        ),
                        className = "row container-fluid"
                    ),
                    className = "page-header"
                )


upload_layout = html.Div([
                    html.Div(
                        html.Div(
                            html.Div(
                                html.H1(
                                    'Please Upload Image to Inpaint',
                                    style = {
                                        'text-align':'center',
                                        'margin' : '40px 0px'
                                })
                            ),
                            className = 'col-md-12 ml-auto mr-auto'
                        ),
                        className = 'row'
                    ),
                    html.Hr(),
                    html.Div(
                        html.Div(
                            dcc.Upload(
                                id = 'upload-image',
                                children = html.Div(
                                                [
                                                    html.P(
                                                        children = 'Drag and Drop or Click to Select Image',
                                                        id = 'upload-text'
                                                    )
                                                ],
                                                className = 'upload-text-wrapper'
                                            ),
                                style = {   
                                    'height': '200px',
                                },
                                style_active = {
                                    'borderColor': 'blue'
                                },
                                accept = '.png, .jpg, .jpeg',
                                # don't allow multiple files to be uploaded
                                multiple = False,
                                className = 'upload-section container-fluid'),
                        className = "col-md-8 ml-auto mr-auto",
                        style = {
                            'display' : 'flex',
                            'justify-content' : 'center',
                        }),
                    id='upload-section-wrapper',
                    className = "row"),
                    html.Div(
                        html.Div(
                            html.A(
                                'Change Image',
                                id = 'change-upload',
                                style = {'display': 'None'}
                            ),
                            className = 'col-md-8 ml-auto mr-auto'
                        ),
                        className = 'row'
                    ),
                    html.Div(
                        html.Div(
                            html.Div(
                                id = 'upload-preview',
                                style = {
                                    'width' : '100%',
                                    'display' : 'flex',
                                    'align-items' : 'center',
                                    'justify-content' : 'center'
                                }
                            ),
                            className = 'col-md-8 ml-auto mr-auto'
                        ),
                        className = 'row'
                    ),
                    html.Div(
                        html.Div(
                            dbc.Button(
                                'Inpaint Image', 
                                color = 'primary',
                                size = 'lg', 
                                className = 'mr-1',
                                id = 'inpaint-button', 
                                block = True,
                                style = {
                                    'margin-top' : '30px',
                                    'font-weight': 'bold',
                                    'padding': '10px'
                                }),
                        className = "col-md-10 ml-auto mr-auto"),
                    className = "row"),
                ])

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 800

def getInpaintLayout(image_content, image_filename) :

    inpaint_layout = html.Div([
                        html.H1(
                            'Draw Mask and Then Run The Inpainter',
                            style = {
                                'text-align':'center',
                                'margin' : '40px 0px'
                        }),
                        html.Div([
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
                        style = {'text-align' : 'center', 'width' : '100%', 'marginTop' : '20px', 'marginBottom' : '20px'}),
                        html.Div([
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
                    ], 
                    className = 'col')
    return inpaint_layout


footer_layout = html.Footer(
                    [
                        html.Ul(
                            [
                                html.Li(html.A("Contact", href="mailto:romdhane.attaher@gmail.com")),
                                html.Li(html.A("Portfolio", href="https://github.com/taherromdhane"))
                            ]
                        ),
                        html.Div(
                            html.Span([
                                "© 2020 Made With ", 
                                html.I(className="fa fa-heart heart"),
                                " & ", 
                                html.I(className="fa fa-coffee"), 
                                " by Tahér & Saif"])
                        )
                    ],
                    className = "page-footer footer-black"
                )