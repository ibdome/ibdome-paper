import dash
from dash import dcc, html
import plotly.graph_objects as go

app = dash.Dash(__name__)

# Define the nodes
labels = [
    "WSI Dataset (1362)",  # 0
    "Scored Images (1212) *",  # 1
    "Modified Naini Cortina Score (699)",  # 2
    "Modified Riley Score (556)",  # 3
    "Berlin cohort (514)",  # 4
    "Erlangen cohort (185)",  # 5
    "Berlin cohort (472)",  # 6
    "Erlangen cohort (84)"  # 7
]

# Define the links between the nodes
source = [0, 1, 1, 2, 2, 3, 3]  # source indices for the links
target = [1, 2, 3, 4, 5, 6, 7]  # target indices for the links
values = [1255, 699, 556, 514, 185, 472, 84]  # values for the flows

# Define colors for the nodes
default_color = "rgba(0, 76, 153, 0.6)"  # Light gray for other nodes
berlin_color = "rgba(255, 178, 102, 0.6)"  
erlangen_color = "rgba(255, 102, 102, 0.6)"
naini_color = "rgba(204, 102, 0, 0.6)"
riley_color = "rgba(204, 0, 0, 0.6)"
filtered_color = "rgba(153, 204, 255, 0.6)"
light_violet_color = "rgba(225,147,211,0.51)"  # Lighter violet for links
# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=30,
        # Removed the second 'line' definition to avoid conflict
        line=dict(color="black", width=0.5),
        label=labels,
        x=[0.1, 0.22, 0.5, 0.5, 1, 1, 1, 1],
        y=[0.0, 0.5, 0.25, 0.78, 0.15, 0.41, 0.745, 0.96],
        color=[
            default_color,  # Total Images
            filtered_color,  # Filtered Images
            naini_color,  # Naini Cortina Score
            riley_color,  # Riley Score
            berlin_color,   # Berlin (Naini)
            erlangen_color,   # Erlangen (Naini)
            berlin_color,   # Berlin (Riley)
            erlangen_color    # Erlangen (Riley)
        ],
    ),
    link=dict(
        source=source,
        target=target,
        value=values,
        color=light_violet_color  # Light violet color for the links
    )
))
fig.write_image("sankey_diagram.svg", format="svg", width=1250, height=450)

# Create the layout with Sankey and Pie charts
app.layout = html.Div([
    html.H1("IBD Histopathology Image Dataset Visualizations"),
    dcc.Graph(
        id='sankey-diagram',
        figure=fig,
        config={'displayModeBar': True}  # Show the mode bar for interaction
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)
