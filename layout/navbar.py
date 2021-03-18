import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

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