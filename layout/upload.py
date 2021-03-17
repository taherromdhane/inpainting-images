import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import glob
import sys



def getUploadLayout(samples_folder) :

    def getCard(source) :
        return dbc.Card(
            html.Div([
                    dbc.CardImg(
                        src = source
                    )
                ],
                className = "sample-img-card",
                id = {
                    'type' : 'sample-img',
                    # 'source' : source.split("\\")[-1]
                    'source' : source
                }
            ),
            color = "light",
            outline = True,
            style = {
                'margin-top': '10%',
                'margin-bottom': '10%'
            }
        )

    samples_layout =  html.Div([
        html.Div(
            html.Div(
                html.H3(
                    'Or select a sample image',
                    style = {
                        'text-align':'center',
                        'margin' : '40px 0px'
                }),
                className = 'col-md-12 ml-auto mr-auto'
            ),
        className = 'row'),
        html.Div(
            html.Div(
                dbc.Row(
                    dbc.Col(
                        dbc.Row(
                            [ 
                                dbc.Col(
                                    getCard(img_source),
                                    className = "col-md-2"
                                ) for img_source in glob.glob(samples_folder[1:] + "/*") 
                            ],
                            style = {
                                'overflowX': 'scroll',
                                'flex-wrap': 'nowrap',
                                'align-items': 'center',
                                'padding': '5px'
                            }
                        ),
                        className = "col-md-10 align-self-center row-eq-height"
                    ),
                    className = "justify-content-center"
                ),
            ),
            className = 'row'
        ),
        html.Div(
            id='sample-image-chosen', 
            style={'display': 'none'}
        )
    ])



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
                            id='upload-time-div', 
                            style={'display': 'none'}
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
                        samples_layout,
                        html.Div(
                            html.Div(
                                [
                                    html.Div(id='upload-alert-div',
                                        style = {
                                            'margin-top' : '30px'}),
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
                                        })
                                ],
                            className = "col-md-10 ml-auto mr-auto"),
                        className = "row"),
                    ])

    return upload_layout