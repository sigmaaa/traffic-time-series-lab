from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
from skimage.restoration import denoise_wavelet
from pyts.decomposition import SingularSpectrumAnalysis

# Load data
df = pd.read_csv("train_ML_IOT.csv", parse_dates=["DateTime"])
df = df[df["Junction"] == 1]
signal = df["Vehicles"].astype(float).values
time = df["DateTime"]

# Options
wavelet_options = ['db4', 'sym5', 'coif3',
                   'bior3.5', 'rbio3.5', 'haar', 'dmey']
threshold_selection_methods = ['VisuShrink', 'BayesShrink']
mode_options = ['soft', 'hard']

# App setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Layout
app.layout = html.Div([
    html.H1(
        'Traffic Time Series Analysis Dashboard',
        style={'textAlign': 'center'}
    ),
    dcc.Tabs(id="tabs", value='tab-1-lab-1', children=[
        dcc.Tab(label='Lab-1', value='tab-1-lab-1'),
        dcc.Tab(label='Lab-2', value='tab-2-lab-2'),
    ]),
    html.Div(id='tab-1-content'),
])

# Tab content rendering


@callback(Output('tab-1-content', 'children'),
          Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1-lab-1':
        return html.Div([
            html.Div([
                html.H2("Wavelet Denoise Options:"),
                html.Label("Select Wavelet:"),
                dcc.Dropdown(
                    id='wavelet-selector',
                    options=[{'label': w, 'value': w}
                             for w in wavelet_options],
                    value='db4',
                    clearable=False,
                    style={'width': '300px'}
                ),
                html.Label("Select Threshold Method:"),
                dcc.Dropdown(
                    id='threshold_method-selector',
                    options=[{'label': m, 'value': m}
                             for m in threshold_selection_methods],
                    value='BayesShrink',
                    clearable=False,
                    style={'width': '300px'}
                ),
                html.Label("Select Mode:"),
                dcc.Dropdown(
                    id='mode-selector',
                    options=[{'label': m, 'value': m} for m in mode_options],
                    value='soft',
                    clearable=False,
                    style={'width': '300px'}
                ),
                html.H2("SSA Options:"),
                html.Label("Select SSA Window:"),
                dcc.Dropdown(
                    id='ssa-window-selector',
                    options=[{'label': i, 'value': i} for i in range(1, 100)],
                    value=6,
                    clearable=False,
                    style={'width': '300px'}
                )
            ], style={'marginBottom': '20px'}),
            dcc.Graph(id='wavelet-denoised-graph')
        ])
    elif tab == 'ttab-2-lab-2':
        return html.Div([
            html.H3('Tab content 2'),
            dcc.Graph(
                id='graph-2-tabs-dcc',
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [5, 10, 6],
                        'type': 'bar'
                    }]
                }
            )
        ])

# Denoising graph callback


@callback(
    Output('wavelet-denoised-graph', 'figure'),
    Input('wavelet-selector', 'value'),
    Input('threshold_method-selector', 'value'),
    Input('mode-selector', 'value'),
    Input('ssa-window-selector', 'value')
)
def update_graph(wavelet, method, mode, window_size):
    denoised_signal = denoise_wavelet(
        signal,
        method=method,
        mode=mode,
        wavelet=wavelet,
        wavelet_levels=None,
        rescale_sigma=False
    )

    ssa = SingularSpectrumAnalysis(window_size)
    components = ssa.fit_transform(signal.reshape(1, -1))

    trend_ssa = components[0][0]
    residual = signal - trend_ssa

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal,
                  mode='lines', name='Original Signal'))
    fig.add_trace(go.Scatter(x=time, y=denoised_signal,
                  mode='lines', name='Denoised Signal'))
    fig.add_trace(go.Scatter(x=time, y=residual,
                  mode='lines', name='SSA residuals'))
    fig.add_trace(go.Scatter(x=time, y=trend_ssa,
                  mode='lines', name='SSA trend'))
    for i in range(components.shape[1]):
        fig.add_trace(go.Scatter(
            x=time, y=components[0, i], mode='lines', name=f"Component {i + 1}"))
    fig.update_layout(
        title="Original Traffic Signal vs Wavelet Denoised vs SSA",
        xaxis_title="Time",
        yaxis_title="Vehicles",
        height=500
    )
    return fig


# Run app
if __name__ == '__main__':
    app.run(debug=True)
