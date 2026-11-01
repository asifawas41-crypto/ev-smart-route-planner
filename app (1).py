# ============================================================
# EV SMART ROUTE PLANNER
# Interactive Multi-Objective Route Optimization
# Gradio 6.0 Compatible
# ============================================================

import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gradio as gr

# ------------------------------------------------------------
# 1. CREATE ROAD NETWORK
# ------------------------------------------------------------

G = nx.Graph()

roads = [
    ("Start", "A", 35, 50),
    ("Start", "B", 45, 60),
    ("A", "B", 20, 40),
    ("A", "C", 40, 55),
    ("B", "C", 25, 45),
    ("B", "D", 45, 60),
    ("C", "D", 30, 50),
    ("C", "Destination", 55, 60),
    ("D", "Destination", 35, 50)
]

for u, v, distance, speed in roads:
    G.add_edge(
        u, v,
        distance=distance,
        travel_time=(distance / speed) * 60
    )

candidate_paths = list(
    nx.all_simple_paths(
        G,
        source="Start",
        target="Destination"
    )
)

# ------------------------------------------------------------
# 2. CHARGING STATION DATA
# ------------------------------------------------------------

station_power = {
    "A": 50,
    "B": 100,
    "C": 60,
    "D": 120
}

# ------------------------------------------------------------
# 3. ROUTE EVALUATION
# ------------------------------------------------------------

def evaluate_route(
    path,
    battery_capacity,
    battery_percent,
    consumption,
    reserve_percent,
    available_stations,
    prices
):

    battery_energy = battery_capacity * battery_percent / 100
    reserve_energy = battery_capacity * reserve_percent / 100

    if battery_energy < reserve_energy:
        return None

    total_distance = 0
    driving_time = 0
    charging_time = 0
    charging_cost = 0
    charging_stops = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        total_distance += G[u][v]["distance"]
        driving_time += G[u][v]["travel_time"]

    for i in range(len(path) - 1):

        current = path[i]
        next_location = path[i + 1]

        next_station_index = len(path) - 1

        for j in range(i + 1, len(path)):
            location = path[j]

            if location in available_stations:
                next_station_index = j
                break

        energy_to_target = 0

        for j in range(i, next_station_index):
            u = path[j]
            v = path[j + 1]

            energy_to_target += (
                G[u][v]["distance"] * consumption
            )

        required_energy = energy_to_target + reserve_energy

        if battery_energy < required_energy:

            if current not in available_stations:
                return None

            if required_energy > battery_capacity:
                return None

            energy_added = required_energy - battery_energy

            charging_cost += energy_added * prices[current]

            charging_time += (
                energy_added / station_power[current]
            ) * 60

            battery_energy += energy_added
            charging_stops.append(current)

        energy_used = (
            G[current][next_location]["distance"] * consumption
        )

        battery_energy -= energy_used

        if battery_energy < reserve_energy - 1e-9:
            return None

    return {
        "Route": " -> ".join(path),
        "Distance (km)": round(total_distance, 2),
        "Driving Time (min)": round(driving_time, 2),
        "Charging Time (min)": round(charging_time, 2),
        "Charging Cost (Rs)": round(charging_cost, 2),
        "Total Time (min)": round(driving_time + charging_time, 2),
        "Charging Stops": ", ".join(charging_stops) or "None"
    }


# ------------------------------------------------------------
# 4. PARETO OPTIMIZATION
# ------------------------------------------------------------

def find_pareto_routes(results):

    if results.empty:
        return results.copy()

    objectives = [
        "Distance (km)",
        "Total Time (min)",
        "Charging Cost (Rs)"
    ]

    values = results[objectives].to_numpy()
    pareto_mask = np.ones(len(results), dtype=bool)

    for i in range(len(values)):
        for j in range(len(values)):

            if i == j:
                continue

            dominates = (
                np.all(values[j] <= values[i])
                and np.any(values[j] < values[i])
            )

            if dominates:
                pareto_mask[i] = False
                break

    return results.loc[pareto_mask].reset_index(drop=True)


# ------------------------------------------------------------
# 5. ROUTE MAP
# ------------------------------------------------------------

positions = nx.spring_layout(G, seed=42)


def create_map(selected_route):

    route_nodes = selected_route["Route"].split(" -> ")

    charging_stops = [
        x.strip()
        for x in selected_route["Charging Stops"].split(",")
        if x.strip()
    ]

    fig = go.Figure()

    # All road connections
    for u, v in G.edges():

        fig.add_trace(go.Scatter(
            x=[positions[u][0], positions[v][0]],
            y=[positions[u][1], positions[v][1]],
            mode="lines",
            line=dict(color="lightgray", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Recommended route
    for i in range(len(route_nodes) - 1):

        u = route_nodes[i]
        v = route_nodes[i + 1]

        fig.add_trace(go.Scatter(
            x=[positions[u][0], positions[v][0]],
            y=[positions[u][1], positions[v][1]],
            mode="lines",
            line=dict(color="red", width=6),
            name="Recommended Route",
            showlegend=(i == 0)
        ))

    # Location markers
    node_colors = []

    for node in G.nodes():

        if node == "Start":
            node_colors.append("blue")
        elif node == "Destination":
            node_colors.append("red")
        elif node in charging_stops:
            node_colors.append("orange")
        else:
            node_colors.append("green")

    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in G.nodes()],
        y=[positions[n][1] for n in G.nodes()],
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=25, color=node_colors),
        name="Locations"
    ))

    fig.update_layout(
        title="Recommended EV Route",
        template="plotly_white",
        height=480,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


# ------------------------------------------------------------
# 6. MAIN OPTIMIZATION FUNCTION
# ------------------------------------------------------------

def plan_ev_route(
    battery_capacity,
    battery_percent,
    consumption,
    reserve_percent,
    priority,
    available_stations,
    price_A,
    price_B,
    price_C,
    price_D
):

    if battery_capacity <= 0:
        return "Battery capacity must be positive.", None, pd.DataFrame()

    if not 0 <= battery_percent <= 100:
        return "Battery percentage must be between 0 and 100.", None, pd.DataFrame()

    if not 0 <= reserve_percent < 100:
        return "Reserve must be between 0 and 100.", None, pd.DataFrame()

    if battery_percent < reserve_percent:
        return "Current battery is below the required reserve.", None, pd.DataFrame()

    if consumption <= 0:
        return "Energy consumption must be positive.", None, pd.DataFrame()

    prices = {
        "A": price_A,
        "B": price_B,
        "C": price_C,
        "D": price_D
    }

    if any(p < 0 for p in prices.values()):
        return "Charging prices cannot be negative.", None, pd.DataFrame()

    available_stations = available_stations or []

    evaluated = []

    for path in candidate_paths:

        result = evaluate_route(
            path,
            battery_capacity,
            battery_percent,
            consumption,
            reserve_percent,
            available_stations,
            prices
        )

        if result is not None:
            evaluated.append(result)

    if not evaluated:
        return (
            "No feasible route found. Enable more charging stations "
            "or increase the battery percentage.",
            None,
            pd.DataFrame()
        )

    results = pd.DataFrame(evaluated)

    pareto = find_pareto_routes(results)

    objectives = {
        "Minimum Distance": "Distance (km)",
        "Minimum Travel Time": "Total Time (min)",
        "Minimum Charging Cost": "Charging Cost (Rs)"
    }

    objective = objectives[priority]

    selected = pareto.sort_values(
        by=[
            objective,
            "Total Time (min)",
            "Distance (km)",
            "Charging Cost (Rs)"
        ]
    ).iloc[0]

    initial_energy = battery_capacity * battery_percent / 100
    reserve_energy = battery_capacity * reserve_percent / 100

    initial_usable_range = max(
        0,
        (initial_energy - reserve_energy) / consumption
    )

    summary = f"""
# ⚡ EV ROUTE RECOMMENDATION

**Selected Priority:** {priority}

## Recommended Route

### {selected['Route']}

| Parameter | Result |
|---|---:|
| Distance | {selected['Distance (km)']} km |
| Driving Time | {selected['Driving Time (min)']} min |
| Charging Time | {selected['Charging Time (min)']} min |
| Charging Cost | ₹{selected['Charging Cost (Rs)']} |
| Total Travel Time | {selected['Total Time (min)']} min |
| Charging Stops | {selected['Charging Stops']} |

## Battery Information

- Battery Capacity: {battery_capacity} kWh
- Initial Battery Energy: {initial_energy:.2f} kWh
- Minimum Reserve: {reserve_energy:.2f} kWh
- Initial Safe Planning Range: {initial_usable_range:.1f} km

## Optimization Summary

- Feasible Routes: {len(results)}
- Pareto-Optimal Routes: {len(pareto)}
- Battery Feasibility: Confirmed by the route simulation
"""

    comparison = pareto[
        [
            "Route",
            "Distance (km)",
            "Total Time (min)",
            "Charging Cost (Rs)",
            "Charging Stops"
        ]
    ].copy()

    route_map = create_map(selected)

    return summary, route_map, comparison


# ------------------------------------------------------------
# 7. INTERACTIVE GRADIO DASHBOARD
# ------------------------------------------------------------

with gr.Blocks() as demo:

    gr.Markdown("""
    # ⚡ EV SMART ROUTE PLANNER

    ### Interactive Multi-Objective EV Route Optimization

    Configure your EV battery, charging stations and route priority.
    The planner recalculates feasible routes based on your inputs.
    """)

    with gr.Row():

        with gr.Column():

            gr.Markdown("## 🔋 EV Battery Configuration")

            battery_capacity = gr.Number(
                value=60,
                label="Battery Capacity (kWh)",
                minimum=0.1
            )

            battery_percent = gr.Slider(
                minimum=0,
                maximum=100,
                value=40,
                step=1,
                label="Current Battery Percentage (%)"
            )

            consumption = gr.Number(
                value=0.18,
                label="Energy Consumption (kWh/km)",
                minimum=0.001
            )

            reserve_percent = gr.Slider(
                minimum=0,
                maximum=50,
                value=10,
                step=1,
                label="Minimum Battery Reserve (%)"
            )

        with gr.Column():

            gr.Markdown("## ⚡ Charging Station Settings")

            available_stations = gr.CheckboxGroup(
                choices=["A", "B", "C", "D"],
                value=["A", "B", "C", "D"],
                label="Available Charging Stations"
            )

            price_A = gr.Number(
                value=12,
                label="Station A Price (Rs/kWh)",
                minimum=0
            )

            price_B = gr.Number(
                value=15,
                label="Station B Price (Rs/kWh)",
                minimum=0
            )

            price_C = gr.Number(
                value=10,
                label="Station C Price (Rs/kWh)",
                minimum=0
            )

            price_D = gr.Number(
                value=14,
                label="Station D Price (Rs/kWh)",
                minimum=0
            )

    gr.Markdown("## 🎯 Route Optimization")

    priority = gr.Radio(
        choices=[
            "Minimum Distance",
            "Minimum Travel Time",
            "Minimum Charging Cost"
        ],
        value="Minimum Travel Time",
        label="Choose Your Optimization Priority"
    )

    calculate_button = gr.Button(
        "🚗 Calculate Best EV Route",
        variant="primary"
    )

    recommendation = gr.Markdown()

    gr.Markdown("## 🗺️ Recommended Route Map")

    route_map = gr.Plot()

    gr.Markdown("## 🏆 Pareto-Optimal Route Comparison")

    comparison_table = gr.Dataframe(
        interactive=False,
        label="Optimized Route Comparison"
    )

    inputs = [
        battery_capacity,
        battery_percent,
        consumption,
        reserve_percent,
        priority,
        available_stations,
        price_A,
        price_B,
        price_C,
        price_D
    ]

    calculate_button.click(
        fn=plan_ev_route,
        inputs=inputs,
        outputs=[
            recommendation,
            route_map,
            comparison_table
        ]
    )

    gr.Markdown("""
    ---
    **Optimization Method:** Pareto Dominance-Based Multi-Objective Optimization

    **Data:** Simulated road network and charging stations.

    **Note:** Charging time assumes constant charging power.
    Real-world traffic and charging-station data are not connected.
    """)


# ------------------------------------------------------------
# 8. LAUNCH DASHBOARD - GRADIO 6.0
# ------------------------------------------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
    debug=False,
    theme=gr.themes.Soft()
)