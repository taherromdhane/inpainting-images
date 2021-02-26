import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

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