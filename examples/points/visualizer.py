import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# init dash app
app = dash.Dash(meta_tags=[{"name": "viewport", "content": "width=device-width", "initial-scale": "1"}])
# needed for hosting
# app = dash.Dash( 
#     __name__,
#     meta_tags=[{"name": "viewport", "content": "width=device-width", "initial-scale": "1"}], 
#     requests_pathname_prefix='/nhl/points-prediction/',
#     title="NHL Points Prediction",
#     update_title="NHL Points Prediction - Loading...",
#     assets_folder="assets",
#     include_assets_files=True,
# )

# set server for hosting
server = app.server

# need for caching data when hosting
# Configure no caching
# @server.after_request
# def add_header(response):
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '0'
#     return response


# load the results.csv
df = pd.read_excel("output/results_current.xlsx")
df_t = pd.read_excel("output_tuned/results_current_tuned.xlsx")

# load the results_extrapolated.csv
df2 = pd.read_excel("output/results_extrapolated.xlsx")
df2_t = pd.read_excel("output_tuned/results_extrapolated_tuned.xlsx")

# load the player_stats.xlsx
df_table = pd.read_excel("output/player_stats.xlsx")

# remove all columns but skater name
df_table = df_table[['skaterFullName']]

# load the player_stats_extrapolated.xlsx
df2_table = pd.read_excel("output/player_stats_extrapolated.xlsx")

# remove all columns but skater name
df2_table = df2_table[['skaterFullName']]

# set table styles
style_cell_conditional=[
    {
        'if': {'column_id': 'Predicted Points (LR)'},
        'backgroundColor': 'blue',
        'color': 'white',
    },
    {
        'if': {'column_id': 'Predicted Points (RF)'},
        'backgroundColor': 'green',
        'color': 'white',
    },
    {
        'if': {'column_id': 'Predicted Points (GB)'},
        'backgroundColor': 'purple',
        'color': 'white',
    },
    {
        'if': {'column_id': 'Actual'},
        'backgroundColor': 'red',
        'color': 'white',
    }
]

app.layout = html.Div([
    html.Div('NHL Top 100 Players (2023/2024) Points Prediction', style={'fontSize': '1.5rem','fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '1rem'}),
    html.Div([
        dcc.Graph(id='player-graph')
    ], style={'width': '100%'}),
    html.Div([
        html.Button('Reset Legend', id='reset-button', style={
                                                                'height': '2rem',
                                                                'backgroundColor': '#007BFF',  # Blue color
                                                                'color': 'white', 
                                                                'border': 'none', 
                                                                'borderRadius': '5px', 
                                                                'padding': '0.5rem',  
                                                                'transition': 'backgroundColor 0.3s ease',  # Smooth color transition
                                                                'cursor': 'pointer',  # Change cursor on hover
                                                                'outline': 'none',  # Remove outline
                                                                'boxShadow': '0px 8px 15px rgba(0, 0, 0, 0.1)'  # Add shadow
                                                            }),
    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'justifyContent': 'flex-end', 'width': '80%'}),
    html.Div([
        html.Div([
                    html.Div('Select a player', style={'fontSize': '1rem','marginRight': '1rem', 'fontWeight': 'bold'}),
        html.Div(
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': f"{i+1}. {name}", 'value': name} for i, name in enumerate(df['Player Name'].unique())],
                value=df['Player Name'].unique()[0],
                placeholder="Select a player",
                ),
                className='select-player-dropdown'
            )
        ], className='select-player-cont'),
        
        html.Div([
            html.Div('Model', style={'fontSize': '1rem','marginRight': '1rem', 'fontWeight': 'bold'}),
        html.Div(
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': 'Default', 'value': 'Default'}, {'label': 'Tuned', 'value': 'Tuned'}],
                value='Default',
                placeholder="Select a model",
                ),
                className='select-model-dropdown'
            ),
    ], className='select-model-cont'),
        
        
    ], className='select-cont'),
    html.Div([
        html.Div('Player Stats (as of Jan 6, 2023)', style={'fontSize': '1.25rem', 'textAlign': 'center', 'marginBottom': '1rem'}),
        dash_table.DataTable(
            id='player-table',
            style_table={
                'whiteSpace': 'normal',
                'height': 'auto', 
                'width':'auto'
            },
            style_cell={'whiteSpace': 'normal', 'height': 'auto'},
            style_cell_conditional = style_cell_conditional
            
        ),
        html.Div('Extrapolated Player Stats (End of 2023/2024 season)', style={'fontSize': '1.25rem',  'textAlign': 'center', 'marginBottom': '1rem', 'marginTop': '1rem'}),
        dash_table.DataTable(
            id='player-table_extrapolated',
            style_cell={'whiteSpace': 'normal', 'height': 'auto'},
            style_cell_conditional= style_cell_conditional
        ),
        html.Div([
            html.Div('Notes:', style={'fontSize': '1.25rem','marginRight': '1rem', 'fontWeight': 'bold'}),
            html.Ul([
                html.Li('"Actual" is the total points for the current season (as of Jan 6, 2024).'),
                html.Li('"Predicted Points" uses the various models (Regression, Random Forest, Gradient Boost) to predict and match player points to the "Actual"'),
                html.Li('The Extrapolated values estimate each players season totals assuming an 82 game season. This is done using the Actual, as well as the 3 models.'),
                html.Li('Click on the legend to hide/show the lines.'),
                html.Li('Click on the lines to see the Points for each model.'),
                html.Li('The "Select a player" dropdown allows you to select a player to view their points prediction.'),
                html.Li('The "Model" dropdown allows you to switch between the default and tuned models.'),
                html.Li('The "Reset Legend" button resets the legend to show all lines.'),
                html.Li('Full source code in the footer below.'),
            ], style={'fontSize': '0.85rem', 'textAlign': 'left', 'padding': '1rem'}),
        ], style={'fontSize': '0.85rem', 'fontStyle': 'italic' ,'textAlign': 'left', 'padding': '1rem'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'space-evenly', 'marginTop': '1rem', 'marginBottom': '1rem'}),
    html.Footer([
        html.Div('Created by: BloodLineAlpha Development', style={'fontSize': '0.75rem', 'textAlign': 'center', 'marginTop': '1rem', 'marginBottom': '1rem'}),
        html.Div([
            html.A(
                'Data Source: https://github.com/bloodlinealpha/scikit-learn-nhl',
                href='https://github.com/bloodlinealpha/scikit-learn-nhl',
                target='_blank'
            )
        ], style={'fontSize': '0.75rem', 'textAlign': 'center', 'marginTop': '1rem', 'marginBottom': '1rem'}),
        html.Div([
            html.A(
                'Contact',
                href='mailto:bloodlinealpha@gmail.com',
                target='_blank'
            ),
            html.Div(' | ', style={'padding': '0 1rem', 'display': 'inline-block'}),
            html.Div('bloodlinealpha@gmail.com', style={'display': 'inline-block'}),
        ], style={'fontSize': '0.75rem', 'textAlign': 'center', 'marginTop': '1rem', 'marginBottom': '1rem'}),

    ], style={'fontSize': '0.75rem', 'textAlign': 'center', 'marginTop': '1rem', 'marginBottom': '1rem'})
    
], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'flex-start', 'fontFamily': 'Arial'})

@app.callback(
    Output('player-graph', 'figure'),
    [
        Input('player-dropdown', 'value'),
        Input('reset-button', 'n_clicks'), 
        Input('model-dropdown', 'value')
    ],
)
def update_graph(selected_dropdown_value, n_clicks, model_dropdown_value):
    if n_clicks is None:
        return create_fig_for_player(selected_dropdown_value, model_dropdown_value)
    else:
        return create_fig_for_player(selected_dropdown_value, model_dropdown_value)

def create_fig_for_player(selected_dropdown_value, model_dropdown_value):
    if model_dropdown_value == 'Default':
        dff = df[df['Player Name'] == selected_dropdown_value]
        dff2 = df2[df2['Player Name'] == selected_dropdown_value]
    else:
        dff = df_t[df_t['Player Name'] == selected_dropdown_value]
        dff2 = df2_t[df2_t['Player Name'] == selected_dropdown_value]

    fig = go.Figure()

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        title={
            'text': "Points Prediction for " + selected_dropdown_value,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
    )

    # Draw solid lines from zero to the actual points
    fig.add_trace(go.Scatter(x=[0, dff['Games Played'].values[0]], y=[0, dff['Actual Points'].values[0]], mode='lines', name='Actual', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[0, dff['Games Played'].values[0]], y=[0, dff['Predicted Points (LR)'].values[0]], mode='lines', name='Predicted Points (LR)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, dff['Games Played'].values[0]], y=[0, dff['Predicted Points (RF)'].values[0]], mode='lines', name='Predicted Points (RF)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[0, dff['Games Played'].values[0]], y=[0, dff['Predicted Points (GB)'].values[0]], mode='lines', name='Predicted Points (GB)', line=dict(color='purple')) )

    # Draw dashed lines from the actual points to the extrapolated points
    fig.add_trace(go.Scatter(x=[dff['Games Played'].values[0], dff2['Games Played'].values[0]], y=[dff['Actual Points'].values[0], dff2['Actual Points'].values[0]], mode='lines', name='Actual Extrapolated', hovertemplate='Actual Extrapolated<extra></extra>', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[dff['Games Played'].values[0], dff2['Games Played'].values[0]], y=[dff['Predicted Points (LR)'].values[0], dff2['Predicted Points (LR)'].values[0]], mode='lines', name='Extrapolated Predicted (LR)', hovertemplate='Extrapolated Predicted (LR)<extra></extra>', line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=[dff['Games Played'].values[0], dff2['Games Played'].values[0]], y=[dff['Predicted Points (RF)'].values[0], dff2['Predicted Points (RF)'].values[0]], mode='lines', name='Extrapolated Predicted (RF)', hovertemplate='Extrapolated Predicted (RF)<extra></extra>', line=dict(dash='dash', color='green')))
    fig.add_trace(go.Scatter(x=[dff['Games Played'].values[0], dff2['Games Played'].values[0]], y=[dff['Predicted Points (GB)'].values[0], dff2['Predicted Points (GB)'].values[0]], mode='lines', name='Extrapolated Predicted (GB)', hovertemplate='Extrapolated Predicted (GB)<extra></extra>', line=dict(dash='dash', color='purple')))

    # Add annotations for the extrapolated points
    fig.add_annotation(
        x=dff['Games Played'].values[0],
        y=dff['Actual Points'].values[0],
        text=str(round(dff['Actual Points'].values[0], 0)) + " points",
        clicktoshow='onoff',
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="red",
        opacity=0.8
    )
    fig.add_annotation(
        x=dff['Games Played'].values[0],
        y=dff['Predicted Points (LR)'].values[0],
        text=str(round(dff['Predicted Points (LR)'].values[0], 0)) + " points",
        clicktoshow='onoff',
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="blue",
        opacity=0.8
    )
    fig.add_annotation(
        x=dff['Games Played'].values[0],
        y=dff['Predicted Points (RF)'].values[0],
        text=str(round(dff['Predicted Points (RF)'].values[0],0)) + " points",
        clicktoshow='onoff',
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="green",
        opacity=0.8
    )
    fig.add_annotation(
        x=dff['Games Played'].values[0],
        y=dff['Predicted Points (GB)'].values[0],
        text=str(round(dff['Predicted Points (GB)'].values[0],0)) + " points",
        clicktoshow='onoff',
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="purple",
        opacity=0.8
    )

    fig.add_annotation(
        x=dff2['Games Played'].values[0], 
        y=dff2['Actual Points'].values[0], 
        text=str(round(dff2['Actual Points'].values[0], 0)) + " points", 
        clicktoshow='onoff', 
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="red",
        opacity=0.8
    )
    fig.add_annotation(
        x=dff2['Games Played'].values[0], 
        y=dff2['Predicted Points (LR)'].values[0], 
        text=str(round(dff2['Predicted Points (LR)'].values[0], 0)) + " points", 
        clicktoshow='onoff', 
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="blue",
        opacity=0.8
        )
    fig.add_annotation(
        x=dff2['Games Played'].values[0], 
        y=dff2['Predicted Points (RF)'].values[0],
            text=str(round(dff2['Predicted Points (RF)'].values[0], 0)) + " points", 
            clicktoshow='onoff', 
            visible=False,
            align="center",
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="green",
            opacity=0.8
        )
    fig.add_annotation(
        x=dff2['Games Played'].values[0], 
        y=dff2['Predicted Points (GB)'].values[0], 
        text=str(round(dff2['Predicted Points (GB)'].values[0], 0)) + " points", 
        clicktoshow='onoff', 
        visible=False,
        align="center",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="purple",
        opacity=0.8
    )

    fig.update_layout(
        xaxis_title='Games Played',
        yaxis_title='Points',
        xaxis=dict(
            range=[0,100],
            tickvals=list(range(0, 101, 10)) + [82] + [dff['Games Played'].values[0]]  # Add tick values from 0 to 100 with a step of 10, and also add 82
        ),
        yaxis=dict(range=[0,200])
    )
    fig.add_shape(
        type="line",
        x0=dff['Games Played'].values[0], y0=0, x1=dff['Games Played'].values[0], y1=1,
        yref="paper",
        line=dict(
            color="red",
            width=3,
        )
    )

    fig.add_shape(
        type="line",
        x0=82, y0=0, x1=82, y1=1,
        yref="paper",
        line=dict(
            dash="dashdot",
            color="red",
            width=3,
        )
    )


    return fig

@app.callback(
    Output('player-table', 'columns'),
    Output('player-table', 'data'),
    [
        Input('player-dropdown', 'value'),
        Input('model-dropdown', 'value')
    ]
    
)
def create_table_for_player(selected_dropdown_value, model_dropdown_value):
    # create a dataframe with the player stats
    dff = df_table[df_table['skaterFullName'] == selected_dropdown_value]
    columns = [{"name": i, "id": i} for i in dff.columns]
    data = dff.round(4).to_dict('records')

    # add model predictions to the column and value
    columns.append({"name": "Predicted Points (LR)", "id": "Predicted Points (LR)"})
    columns.append({"name": "Predicted Points (RF)", "id": "Predicted Points (RF)"})
    columns.append({"name": "Predicted Points (GB)", "id": "Predicted Points (GB)"})
    columns.append({"name": "Actual (Jan 6, 2023)", "id": "Actual"})

    def update_data(dataframe, player_name):
        data = {}
        data["skaterFullName"] = player_name
        data["Predicted Points (LR)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (LR)'].values[0], 0)
        data["Predicted Points (RF)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (RF)'].values[0], 0)
        data["Predicted Points (GB)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (GB)'].values[0], 0)
        data["Actual"] = round(dataframe[dataframe['Player Name'] == player_name]['Actual Points'].values[0], 0)
        
        return data

    if model_dropdown_value == 'Default':
        data[0] = update_data(df, selected_dropdown_value)
    else:
        data[0] = update_data(df_t, selected_dropdown_value)
    
    return columns, data

@app.callback(
    Output('player-table_extrapolated', 'columns'),
    Output('player-table_extrapolated', 'data'),
    [
        Input('player-dropdown', 'value'),
        Input('model-dropdown', 'value')
    ]
)
def create_table_for_player_extrapolated(selected_dropdown_value, model_dropdown_value):
    # create a dataframe with the player stats
    dff = df2_table[df2_table['skaterFullName'] == selected_dropdown_value]
    columns=[{"name": i, "id": i} for i in dff.columns]
    data = dff.round(4).to_dict('records')

    # add extrapolated model predictions to the column and value
    columns.append({"name": "Predicted Points (LR)", "id": "Predicted Points (LR)"})
    columns.append({"name": "Predicted Points (RF)", "id": "Predicted Points (RF)"})
    columns.append({"name": "Predicted Points (GB)", "id": "Predicted Points (GB)"})
    columns.append({"name": "Actual (est.)", "id": "Actual"})

    def update_data(dataframe, player_name):
        data = {}
        data["skaterFullName"] = player_name
        data["Predicted Points (LR)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (LR)'].values[0], 0)
        data["Predicted Points (RF)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (RF)'].values[0], 0)
        data["Predicted Points (GB)"] = round(dataframe[dataframe['Player Name'] == player_name]['Predicted Points (GB)'].values[0], 0)
        data["Actual"] = round(dataframe[dataframe['Player Name'] == player_name]['Actual Points'].values[0], 0)
        return data
    
    if model_dropdown_value == 'Default':
        data[0] = update_data(df2, selected_dropdown_value)
    else:
        data[0] = update_data(df2_t, selected_dropdown_value)

    return columns, data


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True)
    