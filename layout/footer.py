import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

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