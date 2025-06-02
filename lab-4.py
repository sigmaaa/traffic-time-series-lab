import numpy as np
import pandas as pd
import yfinance as yf
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
from sklearn.cluster import KMeans

# Завантаження реального часового ряду


def generate_time_series():
    df = yf.download("AAPL", start="2022-01-01", end="2024-12-31")
    if 'Adj Close' in df.columns:
        series = df['Adj Close'].dropna().values
    else:
        series = df.iloc[:, 0].dropna().values
    return series

# Кластеризація дельт (змін) на k станів


def discretize_changes(series, k):
    deltas = series[1:] - series[:-1]
    deltas_reshaped = deltas.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(deltas_reshaped)
    states = kmeans.predict(deltas_reshaped)
    centers = kmeans.cluster_centers_.flatten()
    return states, centers, deltas

# Обчислення ймовірностей переходу між станами


def estimate_transition_matrix(states, k):
    trans_matrix = np.zeros((k, k))
    for i in range(len(states) - 1):
        trans_matrix[states[i], states[i+1]] += 1
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    return trans_matrix / np.maximum(row_sums, 1e-8)

# Симуляція майбутніх значень


def simulate_piecewise_markov(series, transition_matrix, centers, init_state, n_steps, n_simulations):
    simulations = []
    for _ in range(n_simulations):
        sim = [series[-1]]
        current_state = init_state
        for _ in range(n_steps):
            next_state = np.random.choice(
                len(centers), p=transition_matrix[current_state])
            delta = centers[next_state]
            sim.append(sim[-1] + delta)
            current_state = next_state
        simulations.append(sim)
    return simulations


# Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Piecewise Constant Transition Forecast (AAPL Price)"),
    dcc.Graph(id='forecast-graph'),
    html.Label("Кількість симуляцій:"),
    dcc.Slider(id='sim-count', min=1, max=10, step=1, value=5,
               marks={i: str(i) for i in range(1, 11)}),
    html.Label("Кількість станів (кластерів):"),
    dcc.Slider(id='n-states', min=2, max=6, step=1, value=3,
               marks={i: str(i) for i in range(2, 7)}),
    html.H4("Матриця переходів"),
    dash_table.DataTable(id='transition-matrix',
                         style_table={'overflowX': 'auto'})
])


@app.callback(
    Output('forecast-graph', 'figure'),
    Output('transition-matrix', 'data'),
    Output('transition-matrix', 'columns'),
    Input('sim-count', 'value'),
    Input('n-states', 'value')
)
def update_forecast(sim_count, n_states):
    series = generate_time_series()
    states, centers, deltas = discretize_changes(series, n_states)
    transition_matrix = estimate_transition_matrix(states, n_states)
    init_state = states[-1]
    simulations = simulate_piecewise_markov(
        series, transition_matrix, centers, init_state, 1000, sim_count)

    traces = [go.Scatter(y=series, name="Original Series")]
    for i, sim in enumerate(simulations):
        traces.append(go.Scatter(
            x=np.arange(len(series), len(series) + len(sim)),
            y=sim,
            name=f"Forecast {i+1}"
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Markov Simulation with Piecewise Transition Intensities")

    df_matrix = pd.DataFrame(transition_matrix, columns=[
                             f"S{j}" for j in range(n_states)])
    df_matrix.insert(0, "From/To", [f"S{i}" for i in range(n_states)])
    columns = [{"name": col, "id": col} for col in df_matrix.columns]

    return fig, df_matrix.to_dict("records"), columns


if __name__ == '__main__':
    app.run(debug=False)
