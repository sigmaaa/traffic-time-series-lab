from dash import Dash, dcc, html, Input, Output, State, callback
import pandas as pd
import plotly.graph_objs as go
from skimage.restoration import denoise_wavelet
from pyts.decomposition import SingularSpectrumAnalysis
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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


def preprocess_signal(df, junction=1):
    df = df[df['Junction'] == junction].copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    return df[['DateTime', 'Vehicles']].reset_index(drop=True)


def quantize_signal(signal, n_states):
    kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=0)
    states = kmeans.fit_predict(signal.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    return states, centers


def build_transition_matrix(states, n_states):
    trans_matrix = np.zeros((n_states, n_states))
    for (i, j) in zip(states[:-1], states[1:]):
        trans_matrix[i, j] += 1
    trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
    return np.nan_to_num(trans_matrix)


def em_transition_model(states, n_states):
    X = np.column_stack([states[:-1], states[1:]])
    gmm = GaussianMixture(n_components=n_states,
                          covariance_type='full', random_state=0)
    gmm.fit(X)
    return gmm


def predict_markov_chain(current_state, trans_matrix, steps):
    n_states = trans_matrix.shape[0]
    probs = np.zeros((steps + 1, n_states))
    probs[0, current_state] = 1.0
    for t in range(1, steps + 1):
        probs[t] = probs[t - 1] @ trans_matrix
    return probs


# Function to compute W-correlation matrix


# --- Графічний метод: відстань від прямої ---
def compute_distances(t, z_smooth, m, n):
    tm, tn = t[m], t[n]
    zm, zn = z_smooth[m], z_smooth[n]

    # Параметри прямої: z(t) = kt + b
    k = (zn - zm) / (tn - tm)
    b = zm - k * tm

    # Обчислюємо d_ti
    d_ti = np.abs(k * t[m:n+1] - z_smooth[m:n+1] + b) / np.sqrt(k**2 + 1)
    return d_ti, k, b

# --- Рекурсивне виявлення змін + U-критерій --


def find_change_point(t, z_smooth, m, n, bar, change_points):
    if n - m < 25:  # мінімальна довжина сегмента
        return

    d_ti, k, b = compute_distances(t, z_smooth, m, n)
    idx_local = np.argmax(d_ti)
    global_idx = m + idx_local
    d_max = d_ti[idx_local]

    # U-критерій між лівим і правим сегментом
    left = z_smooth[m:global_idx]
    right = z_smooth[global_idx:n]

    if len(left) < 10 or len(right) < 10:
        return

    u_stat, p_value = mannwhitneyu(left, right, alternative='two-sided')

    if d_max > bar and p_value < 0.05:
        change_points.append(global_idx)
    find_change_point(t, z_smooth,  m, global_idx, bar, change_points)
    find_change_point(t, z_smooth,  global_idx, n, bar, change_points)


def detect_changes(t, z_smooth, bar=2.0):
    change_points = []
    find_change_point(t, z_smooth, 0, len(t) - 1, bar, change_points)
    return change_points


def w_correlation_matrix(components):
    num_components = components.shape[1]
    W = np.zeros((num_components, num_components))
    for i in range(num_components):
        for j in range(num_components):
            num = np.dot(components[0, i], components[0, j])
            den = np.sqrt(np.dot(
                components[0, i], components[0, i]) * np.dot(components[0, j], components[0, j]))
            W[i, j] = num / den if den != 0 else 0
    return W


# Layout
app.layout = html.Div([
    html.H1(
        'Traffic Time Series Analysis Dashboard',
        style={'textAlign': 'center'}
    ),
    dcc.Tabs(id="tabs", value='tab-1-lab-1', children=[
        dcc.Tab(label='Lab-1', value='tab-1-lab-1'),
        dcc.Tab(label='Lab-2', value='tab-2-lab-2'),
        dcc.Tab(label="Lab-3: Markov Forecast", children=[
            html.Div([
                html.H4(
                    "Короткочасний прогноз трафіку за допомогою ланцюгів Маркова"),
                dcc.Slider(id='lab3-k-slider', min=2, max=10, step=1, value=4,
                           marks={i: str(i) for i in range(2, 11)},
                           tooltip={"placement": "bottom"}),
                html.Label("Горизонт прогнозування (год):"),
                dcc.Input(id='lab3-horizon', type='number',
                          value=6, min=1, max=24),
                html.Button("Прогнозувати",
                            id='lab3-forecast-button', n_clicks=0),
                dcc.Graph(id='lab3-prediction-graph')
            ])
        ]),
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
            dcc.Graph(id='wavelet-denoised-graph'),
            dcc.Graph(id='w-correlation-graph')
        ])
    elif tab == 'tab-2-lab-2':
        return html.Div([
            html.H3("Change Point Detection & Mann–Whitney U Test (denoised by SSA)"),
            html.Label("Threshold (bar) for detecting disorder:"),
            dcc.Slider(
                id='bar-slider',
                min=2,
                max=150,
                step=1,
                value=60,
                marks={i: str(i) for i in range(10, 151, 20)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag',
            ),
            dcc.Graph(id='residuals-graph'),
        ])


# Main callback


@callback(
    Output('wavelet-denoised-graph', 'figure'),
    Output('w-correlation-graph', 'figure'),
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

    trend_ssa = np.sum(components[0][:3], axis=0)
    residual = signal - trend_ssa

    # Main graph
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

    # W-correlation matrix
    W = w_correlation_matrix(components)
    heatmap = go.Figure(data=go.Heatmap(
        z=W,
        x=[f"C{i+1}" for i in range(W.shape[0])],
        y=[f"C{i+1}" for i in range(W.shape[0])],
        colorscale='Viridis'
    ))
    heatmap.update_layout(
        title="W-Correlation Map of SSA Components",
        xaxis_title="Components",
        yaxis_title="Components",
        height=500
    )

    return fig, heatmap


@callback(
    Output('residuals-graph', 'figure'),
    Input('tabs', 'value'),
    Input('bar-slider', 'value')
)
def update_tab2(tab, bar):
    if tab != 'tab-2-lab-2':
        return go.Figure(), ""
    # SSA з вікном 6
    window_size = 6
    ssa = SingularSpectrumAnalysis(window_size)
    components = ssa.fit_transform(signal.reshape(1, -1))
    trend_ssa = components[0][0]
    denoised_signal = trend_ssa

    # Графічний апост метод
    numeric_time = np.arange(len(trend_ssa))
    change_points_idxs = detect_changes(numeric_time, denoised_signal, bar)

    # Точки часу, коли сталося розладнання
    change_points_times = [time[i] for i in change_points_idxs]

    # Побудова графіка
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=denoised_signal,
                             mode='lines', name='Denoised signal'))

    # Додаємо вертикальні лінії
    for t in change_points_times:
        fig.add_shape(
            type='line',
            x0=t,
            x1=t,
            y0=min(denoised_signal),
            y1=max(denoised_signal),
            line=dict(color='orange', width=2, dash='dash')
        )

    # Додаємо "примарну" лінію для легенди
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name='Change Points'
    ))

    fig.update_layout(
        title="Outliners detection on Denoised Signal",
        xaxis_title="Time",
        yaxis_title="Vehicles",
        height=500
    )

    return fig


@app.callback(
    Output('lab3-prediction-graph', 'figure'),
    Input('lab3-forecast-button', 'n_clicks'),
    State('lab3-k-slider', 'value'),
    State('lab3-horizon', 'value')
)
def update_lab3_forecast(n_clicks, k, horizon):
    if n_clicks == 0:
        return go.Figure()

    df = pd.read_csv('train_ML_IOT.csv')
    signal_df = preprocess_signal(df)
    signal = signal_df['Vehicles'].values
    states, centers = quantize_signal(signal, k)
    transition_matrix = build_transition_matrix(states, k)

    # Поточний стан — останній у сигналі
    current_state = states[-1]

    # Прогноз на horizon кроків
    probs = predict_markov_chain(current_state, transition_matrix, horizon)

    fig = go.Figure()
    for s in range(k):
        fig.add_trace(go.Scatter(
            x=list(range(horizon + 1)),
            y=probs[:, s],
            mode='lines+markers',
            name=f"Стан {s} (центр = {centers[s]:.1f})"
        ))

    fig.update_layout(title="Ймовірності станів у майбутньому",
                      xaxis_title="Крок прогнозу",
                      yaxis_title="Ймовірність",
                      yaxis=dict(range=[0, 1]))
    return fig


# Run app
if __name__ == '__main__':
    app.run(debug=False)
