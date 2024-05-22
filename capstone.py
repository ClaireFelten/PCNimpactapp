import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# AN APP BY CLAIRE FELTEN 

# Load the dataset
data = pd.read_csv('/Users/clairefeltenpath/Desktop/PCN county data.csv')
data['PCNs_all'] = data['PCNs_not'] + data['PCNs_inprog'] + data['PCNs_complete']
data['PCNs_not_perc'] = data['PCNs_not'] / data['PCNs_all']
data['PCNs_inprog_perc'] = data['PCNs_inprog'] / data['PCNs_all']
data['PCNs_complete_perc'] = data['PCNs_complete'] / data['PCNs_all']

county_info_df = pd.read_csv('/Users/clairefeltenpath/Desktop/county_info.csv')
county_sizepop_df = pd.read_csv('/Users/clairefeltenpath/Desktop/county_sizepop.csv')


# Convert non-numeric values to NaN
county_sizepop_df['Total'] = pd.to_numeric(county_sizepop_df['Total'], errors='coerce')
county_sizepop_df['Male'] = pd.to_numeric(county_sizepop_df['Male'], errors='coerce')
county_sizepop_df['Female'] = pd.to_numeric(county_sizepop_df['Female'], errors='coerce')
county_sizepop_df['SQ.Km'] = pd.to_numeric(county_sizepop_df['SQ.Km'], errors='coerce')

# Drop NaN values:
county_sizepop_df = county_sizepop_df.dropna(subset=['Total','Male','Female','SQ.Km'])

# Group by 'County' and sum the specified columns
grouped_df = county_sizepop_df.groupby('County', dropna=True).agg({
    'Total': 'sum',
    'Male': 'sum',
    'Female': 'sum',
    'SQ.Km': 'sum'
}).rename(columns={
    'Total': 'tot_pop', 
    'Male': 'm_pop',
    'Female': 'f_pop',
    'SQ.Km': 'tot_area'
})

grouped_df['pop_density'] = grouped_df['tot_pop'] / grouped_df['tot_area']

# Merge 'data' and 'county_info_df' on 'County'
merge1_df = pd.merge(data, county_info_df, on='County', how='left')

# Now merge the resulting DataFrame with 'county_sizepop_df' on 'County'
final_data = pd.merge(merge1_df, grouped_df, on='County', how='left')


final_data['num_WCBA'] = (final_data['f_pop'] * (final_data['perc_pop_15_49']/100))
final_data['num_births'] = final_data['TFR15_49'] * final_data['num_WCBA']
final_data['GCP'] = pd.to_numeric(final_data['GCP'], errors='coerce')
final_data['GCPpc'] = final_data['GCP']/final_data['tot_pop']

max_GCPpc = final_data['GCPpc'].max()
max_pop_density = final_data['pop_density'].max()

# Split the data into features and target variables
X = final_data[['PCNs_inprog_perc', 'PCNs_complete_perc', 'GCPpc', 'wealth_quint5','TFR15_49','pop_density','perc_pop_15_49']].values
y = final_data[['MMR','U1VAX']].values

# Handle NaN values
X_imputer = SimpleImputer(strategy='median')
y_imputer = SimpleImputer(strategy='median')
X = X_imputer.fit_transform(X)
y = y_imputer.fit_transform(y)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Model - MLPRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
mlp_model = MLPRegressor(hidden_layer_sizes=(8,), solver='adam', max_iter=1200, random_state=42)

rf_model.fit(X, y)
mlp_model.fit(X, y)

# Get the predictions from the model
rf_predictions = rf_model.predict(X)
mlp_predictions = mlp_model.predict(X)
ensemble_predictions = (rf_predictions + mlp_predictions) / 2

#mlp_predictions_original = scaler_y.inverse_transform(mlp_predictions) #added this line
ensemble_predictions_original = scaler_y.inverse_transform(ensemble_predictions)

# Inverse transform the scaled predictions to the original scale
y_original = scaler_y.inverse_transform(y)

# diagnostics
overall_r2 = r2_score(y_original, ensemble_predictions_original) #ensemble
overall_mse = mean_squared_error(y_original, ensemble_predictions_original) #and again

# Calculate R^2 and MSE for each target variable
metrics = {}
for i, target in enumerate(['MMR','U1VAX']):
    r2 = r2_score(y[:, i], ensemble_predictions[:, i]) 
    mse = mean_squared_error(y[:, i], ensemble_predictions[:, i])
    metrics[target] = {'R2': r2, 'MSE': mse}

# Set up the Streamlit app
st.set_page_config(layout="wide")
st.title("Establishment of Primary Care Networks in Kenya: county health outcomes simulator")
st.divider()

col1a, col2a = st.columns([1,2])
with col1a:
   # Interactive sliders and selectors remain mostly the same
    st.subheader("First, choose a county:")
    selected_county = st.selectbox("County", final_data['County'].unique())
    county_data = final_data[final_data['County'] == selected_county].reset_index(drop=True)

    # Get the row data for the selected county
    county_data = final_data[final_data['County'] == selected_county].reset_index(drop=True)
    
    # Show PCNs status with a progress bar
    pcns_not = int(county_data['PCNs_not'][0])
    pcns_inprog = int(county_data['PCNs_inprog'][0])
    pcns_complete = int(county_data['PCNs_complete'][0])
    
    #st.subheader("PCNs Status")
    pcns_total = pcns_not + pcns_inprog + pcns_complete
    pcns_not_perc = pcns_not / pcns_total * 100
    pcns_inprog_perc = pcns_inprog / pcns_total * 100
    pcns_complete_perc = pcns_complete / pcns_total * 100
    
    # Create a slider for PCNs complete
    max_total_pcns = pcns_total
    st.divider()
    st.subheader("Then move the slider to simulate the establishment of additional PCNs in your county - and see the estimated public health impact of investing in PHC:")
    pcns_complete_slider = st.slider("\# of PCNs established", 0, int(max_total_pcns), pcns_complete)

with col2a:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['PCNs status'],
        x=[pcns_not_perc],
        name='Not started',
        orientation='h',
        marker=dict(
            color=['red'],
            line=dict(color='rgba(246, 78, 139, 1.0)', width=.5)
        ),
        text=[str(pcns_not)+' PCNs not yet established'],  # Add data label for 'Not started'
        textposition='auto',
        textfont=dict(
            size=18,
            color='white' 
        ),
        insidetextanchor='middle'
    ))

    fig.add_trace(go.Bar(
        y=['PCNs status'],
        x=[pcns_inprog_perc],
        name='In progress',
        orientation='h',
        marker=dict(
            color=['yellow'],
            line=dict(color='rgba(246, 78, 139, 1.0)', width=.5)
        ),
        text=[str(pcns_inprog)+' in progress'],  # Add data label for 'In progress'
        textposition='auto',
        textfont=dict(
            size=18,
            color='black' 
        ),
        insidetextanchor='middle'
    ))

    fig.add_trace(go.Bar(
        y=['PCNs status'],
        x=[pcns_complete_perc],
        name='Complete',
        orientation='h',
        marker=dict(
            color=['green'],
            line=dict(color='grey', width=.5)
        ),
        text=[str(pcns_complete) + ' PCNs fully established'],  # Add data label for 'Complete'
        textposition='auto',
        textfont=dict(
            size=18,
            color='white' 
        ),
        insidetextanchor='middle'
    ))
    fig.update_traces(width=.3)
    fig.update_layout(barmode='stack')
    st.plotly_chart(fig,use_container_width=True)

posorneg = 'zero'   

# Adjust PCNs in progress and not started based on the slider
if pcns_complete_slider > pcns_complete:
    posorneg = 'pos'
    additional_complete = pcns_complete_slider - pcns_complete
    if additional_complete <= pcns_inprog:
        pcns_inprog -= additional_complete
    else:
        remaining = additional_complete - pcns_inprog
        pcns_inprog = 0
        pcns_not -= remaining
elif pcns_complete_slider < pcns_complete:
    posorneg = 'neg'
    reduction_complete = pcns_complete - pcns_complete_slider
    if reduction_complete <= pcns_not:
        pcns_not += reduction_complete
    else:
        remaining = reduction_complete - pcns_not
        pcns_not = 0
        pcns_inprog += remaining

county_GCPpc = county_data.loc[0, 'GCPpc'] / max_GCPpc
county_wealthquint5 = county_data.loc[0, 'wealth_quint5']
county_TFR = county_data.loc[0, 'TFR15_49']
county_pop_density = county_data.loc[0, 'pop_density'] / max_pop_density
county_pop_young = county_data.loc[0, 'perc_pop_15_49']

# Create an input row with the original PCN values
original_input = [county_data.loc[0, 'PCNs_inprog'] / max_total_pcns,
                  county_data.loc[0, 'PCNs_complete'] / max_total_pcns,
                  county_GCPpc,
                  county_wealthquint5,
                  county_TFR,
                  county_pop_density,
                  county_pop_young
                  ]

# Normalize the original input
original_input_scaled = scaler_X.transform([original_input])

# Make predictions using the model for the original input
original_predictions_scaled = (rf_model.predict(original_input_scaled) + mlp_model.predict(original_input_scaled)) / 2
original_predictions = scaler_y.inverse_transform(original_predictions_scaled)

# Create a new input row with the updated values
new_input = [pcns_inprog / max_total_pcns, pcns_complete_slider / max_total_pcns,
             county_GCPpc, county_wealthquint5, county_TFR, county_pop_density, county_pop_young]

# Normalize the new input
new_input_scaled = scaler_X.transform([new_input])

# Make predictions using the model for the new input
new_predictions_scaled = (rf_model.predict(new_input_scaled) + mlp_model.predict(new_input_scaled)) / 2
new_predictions = scaler_y.inverse_transform(new_predictions_scaled)

# Calculate the change in values
mmr_change = new_predictions[0][0] - original_predictions[0][0]
u1vax_change = new_predictions[0][1] - original_predictions[0][1]

# Set the y-axis range for both plots
max_mmr = max(original_predictions[0][0], new_predictions[0][0]) * 1.1
max_u1vax = max(original_predictions[0][1], new_predictions[0][1]) * 1.1

# number of mothers lives saved = MMR per 100000 / 100000 * num of births
moms_saved = abs(round(mmr_change/100000 * county_data.loc[0,'num_births']))
if mmr_change > 0:
    mmr_sign = 'increase'
elif mmr_change < 0:
    mmr_sign = 'decrease'
else:
    mmr_sign = '(no change)'
    
kids_vaxxed = abs(round(u1vax_change/100*county_data.loc[0,'num_births']))
if u1vax_change < 0:
    u1vax_sign = 'decrease'
elif u1vax_change > 0:
    u1vax_sign= 'increase'
else:
    u1vax_sign = '(no change)'

# Plot MMR line graph
fig_mmr = go.Figure()
fig_mmr.add_trace(go.Scatter(x=['Current stat', 'Simulated outcome'], y=[original_predictions[0][0], new_predictions[0][0]],
                             mode='lines+markers',
                             marker=dict(color=['gray', 'blue'], size=15),
                             line=dict(color='royalblue'),
                             text=[f"{original_predictions[0][0]:.2f}", f"{new_predictions[0][0]:.2f}"],
                             textposition='top center'))
fig_mmr.update_layout(title=f'Maternal Mortality Ratio (MMR): [{mmr_change:.2f}] (per 100,000) {mmr_sign}',
                      xaxis_title='Scenario',
                      yaxis_title='MMR (per 100,000 live births)',
                      yaxis_range=[0, max_mmr],
                      showlegend=False)

# Add annotation for change and arrow
if mmr_change >= 0:
    fig_mmr.add_annotation(text=f"[{mmr_change:.2f}] increase",
                           xref="paper", yref="paper",
                           x=1.1, y=1.05, showarrow=False)
    fig_mmr.add_annotation(x=[0.5, 1.5], y=[original_predictions[0][0], new_predictions[0][0]],
                           ax=0, ay=60, arrowcolor="green", arrowsize=2, arrowwidth=2, arrowhead=3)
else:
    fig_mmr.add_annotation(text=f"[{-mmr_change:.2f}] decrease",
                           xref="paper", yref="paper",
                           x=1.1, y=1.05, showarrow=False)
    fig_mmr.add_annotation(x=[0.5, 1.5], y=[original_predictions[0][0], new_predictions[0][0]],
                           ax=0, ay=-60, arrowcolor="red", arrowsize=2, arrowwidth=2, arrowhead=3)

# Plot U1VAX line graph
fig_u1vax = go.Figure()
fig_u1vax.add_trace(go.Scatter(x=['Current stat', 'Simulated outcome'], y=[original_predictions[0][1], new_predictions[0][1]],
                               mode='lines+markers',
                               marker=dict(color=['gray', 'blue'], size=15),
                               line=dict(color='royalblue'),
                               text=[f"{original_predictions[0][1]:.2f}", f"{new_predictions[0][1]:.2f}"],
                               textposition='top center'))
fig_u1vax.update_layout(title=f'Proportion of Children <1 Year Who Received DTP3 Vaccine: [{u1vax_change:.2f}] percentage points {u1vax_sign}',
                        xaxis_title='Scenario',
                        yaxis_title='Proportion',
                        yaxis_range=[0, max_u1vax],
                        showlegend=False)

# Add annotation for change and arrow
if u1vax_change >= 0:
    fig_u1vax.add_annotation(text=f"[{u1vax_change:.2f}] increase",
                             xref="paper", yref="paper",
                             x=1.1, y=1.05, showarrow=False)
    fig_u1vax.add_annotation(x=[0.5, 1.5], y=[original_predictions[0][1], new_predictions[0][1]],
                             ax=0, ay=60, arrowcolor="green", arrowsize=2, arrowwidth=2, arrowhead=3)
else:
    fig_u1vax.add_annotation(text=f"[{-u1vax_change:.2f}] decrease",
                             xref="paper", yref="paper",
                             x=1.1, y=1.05, showarrow=False)
    fig_u1vax.add_annotation(x=[0.5, 1.5], y=[original_predictions[0][1], new_predictions[0][1]],
                             ax=0, ay=-60, arrowcolor="red", arrowsize=2, arrowwidth=2, arrowhead=3)

st.divider()


if county_data.loc[0, 'PCNs_not'] <= 0:
    st.subheader('Congrats! Your county has fully established all its PCNs.')
elif posorneg == 'pos':
    st.subheader(f'Predicted public health impact of establishing {additional_complete} additional PCN(s)')
elif posorneg == 'neg':
    st.subheader(f'Predicted public health impact of removing {reduction_complete} PCN(s)')
else:
    st.subheader(f'Public health impact: choose a scenario to compare')




col1, col2 = st.columns(2)
with col1:
    if mmr_change > 0:
        if pcns_complete_slider < pcns_complete:
            st.subheader(f' = {moms_saved} additional maternal deaths')
            st.divider()

    elif mmr_change < 0:
        st.subheader(f' = {moms_saved} mothers\' lives saved')   
        st.divider()     
        
    st.plotly_chart(fig_mmr)
with col2:
    if u1vax_change > 0:
        st.subheader(f' = {kids_vaxxed} additional kids vaccinated')
        st.divider()
    
    elif u1vax_change < 0:
        if pcns_complete_slider < pcns_complete:
            st.subheader(f' = {kids_vaxxed} kids missing their vaccines and exposed to greater risk of disease')   
            st.divider()     
    
    st.plotly_chart(fig_u1vax,use_container_width=True)
