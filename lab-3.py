import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Define states
states = [0, 1, 2]
state_names = {0: "Low", 1: "Medium", 2: "High"}
n_states = len(states)


def simulate_ctmc(Q, t_max, initial_state=0):
    t = 0
    t_series = [t]
    state_series = [initial_state]

    current_state = initial_state

    while t < t_max:
        rate = -Q[current_state, current_state]
        if rate == 0:
            break
        wait_time = np.random.exponential(1 / rate)
        t += wait_time
        probs = Q[current_state].copy()
        probs[current_state] = 0
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            break
        probs /= probs.sum()
        next_state = np.random.choice(states, p=probs)

        t_series.append(t)
        state_series.append(next_state)
        current_state = next_state

    return np.array(t_series), np.array(state_series)


# Base generator matrix Q (constant)
Q_base = np.array([
    [-0.3,  0.2,  0.1],
    [0.1, -0.4,  0.3],
    [0.05, 0.2, -0.25]
])

# Generate time points for spline approx
Q_times = np.linspace(0, 100, 20)

# Create splines for time-varying intensities


def create_splines(Q_base, Q_times):
    Q_t_list = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            if i == j:
                row.append(None)
            else:
                # Example time-varying intensity: base * (1 + 0.5*sin(...))
                rate_t = Q_base[i, j] * \
                    (1 + 0.5 * np.sin(Q_times * 2 * np.pi / 100))
                spline = CubicSpline(Q_times, rate_t)
                row.append(spline)
        Q_t_list.append(row)
    return Q_t_list


Q_t_list = create_splines(Q_base, Q_times)


def get_Q_t(t):
    Q_t = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                val = Q_t_list[i][j](t)
                Q_t[i, j] = max(val, 0)  # ensure no negative rates
        Q_t[i, i] = -np.sum(Q_t[i, :])
    return Q_t


def forward_equation(t, P_flat):
    P = P_flat.reshape((1, n_states))
    Q_t = get_Q_t(t)
    dPdt = P @ Q_t
    return dPdt.flatten()


# Layout of the app
app.layout = html.Div([
    html.H1("CTMC Traffic Series Simulation & Forecast with Spline Approximation"),

    html.Div([
        html.Label("Simulation time (max t):"),
        dcc.Slider(id='t_max-slider', min=50, max=200, step=10, value=100,
                   marks={i: str(i) for i in range(50, 201, 50)})
    ], style={'width': '50%', 'padding': '20px'}),

    html.Button('Simulate & Forecast', id='simulate-btn', n_clicks=0),

    dcc.Graph(id='traffic-trajectory'),
    dcc.Graph(id='forecast-probabilities'),
])


@app.callback(
    [Output('traffic-trajectory', 'figure'),
     Output('forecast-probabilities', 'figure')],
    Input('simulate-btn', 'n_clicks'),
    Input('t_max-slider', 'value')
)
def update_simulation(n_clicks, t_max):
    if n_clicks == 0:
        fig1 = go.Figure()
        fig1.update_layout(title="Traffic State Trajectory")
        fig2 = go.Figure()
        fig2.update_layout(title="Forecasted State Probabilities")
        return fig1, fig2

    # Simulate traffic states
    t_series, state_series = simulate_ctmc(Q_base, t_max)

    # Forecast from last observed state for 20 time units
    initial_prob = np.zeros(n_states)
    initial_prob[state_series[-1]] = 1.0
    t_forecast = np.linspace(t_series[-1], t_series[-1] + 20, 200)
    sol = solve_ivp(forward_equation, [t_forecast[0], t_forecast[-1]], initial_prob,
                    t_eval=t_forecast, vectorized=True)

    # Plot trajectory: step plot of states
    trajectory_fig = go.Figure()
    trajectory_fig.add_trace(go.Scatter(
        x=t_series,
        y=state_series,
        mode='lines+markers',
        line=dict(shape='hv'),
        marker=dict(size=6),
        name="Observed State"
    ))

    # Add forecasted states as green markers at each forecast time point
    forecast_states = np.argmax(sol.y, axis=0)  # most probable states
    trajectory_fig.add_trace(go.Scatter(
        x=t_forecast,
        y=forecast_states,
        mode='lines+markers',
        marker=dict(color='green', size=6),
        name="Forecasted State"
    ))

    trajectory_fig.update_yaxes(tickvals=states, ticktext=[
                                state_names[s] for s in states])
    trajectory_fig.update_layout(title="Simulated Traffic State Trajectory + Forecasted States",
                                 xaxis_title="Time",
                                 yaxis_title="Traffic State")

    # Plot forecasted probabilities
    forecast_fig = go.Figure()
    for i in range(n_states):
        forecast_fig.add_trace(go.Scatter(
            x=sol.t,
            y=sol.y[i],
            mode='lines',
            name=f"State {state_names[i]}"
        ))
    forecast_fig.update_layout(title="Forecasted State Probabilities",
                               xaxis_title="Time",
                               yaxis_title="Probability",
                               yaxis=dict(range=[0, 1]))

    return trajectory_fig, forecast_fig


if __name__ == '__main__':
    app.run(debug=True)
