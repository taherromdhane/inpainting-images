import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

signature_layout = html.Div(
                            html.Span([
                                "© 2020 Made With ", 
                                html.I(className="fa fa-heart heart"),
                                " & ", 
                                html.I(className="fa fa-coffee"), 
                                " by Tahér & Saif"])
                        )

links_layout = html.Ul(
                    [
                        html.Li(html.A("Contact", href="mailto:romdhane.attaher@gmail.com")),
                        html.Li(html.A("Portfolio", href="https://github.com/taherromdhane"))
                    ]
                )

footer_layout = html.Footer(
                    [
                        links_layout,
                        signature_layout
                    ],
                    className = "page-footer footer-black"
                )