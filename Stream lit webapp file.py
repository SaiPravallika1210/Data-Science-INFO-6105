import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
 
 
# Read data
races = pd.read_csv('sample_data/races.csv')
circuits = pd.read_csv('sample_data/circuits.csv')
drivers = pd.read_csv('sample_data/drivers.csv')
drivers_standing = pd.read_csv('sample_data/driver_standings.csv')
constructors = pd.read_csv('sample_data/constructors.csv')
constructor_standings = pd.read_csv('sample_data/constructor_standings.csv')
constructor_results = pd.read_csv('sample_data/constructor_results.csv')
 
# Data preprocessing
columns_to_drop_races = ["time", "url", "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
                         "quali_date", "quali_time", "sprint_date", "sprint_time"]
df_races = races.drop(columns=columns_to_drop_races, axis=1).rename(columns={'round': 'race_YTD'})
 
columns_to_drop_circuits = ["circuitRef", "alt", "url"]
df_circuits = circuits.drop(columns=columns_to_drop_circuits, axis=1)
 
columns_to_drop_drivers = ["driverRef", "number", "code", "url"]
df_drivers = drivers.drop(columns=columns_to_drop_drivers, axis=1)
 
columns_to_drop_constructors = ["constructorRef", "url"]
df_constructors = constructors.drop(columns=columns_to_drop_constructors, axis=1).rename(columns={'name': 'Team_name'})
 
columns_to_drop_constructor_results = ["status"]
df_constructor_results = constructor_results.drop(columns=columns_to_drop_constructor_results, axis=1)
 
df_drivers_standing = drivers_standing
 
df_constructor_standings = constructor_standings
# Merge dataframes
merged_constructors = pd.merge(pd.merge(df_constructor_standings, df_constructors, on='constructorId'),
                               df_races, on='raceId')
df_merged_constructors = pd.DataFrame(merged_constructors)
 
# Streamlit app
st.title("Formula 1 Data Explorer")
 
# Display dataframes
st.header("Races Data")
st.dataframe(df_races)
 
st.header("Circuits Data")
st.dataframe(df_circuits)
 
st.header("Drivers Data")
st.dataframe(df_drivers)
 
st.header("Driver Standings Data")
st.dataframe(df_drivers_standing)
 
st.header("Constructors Data")
st.dataframe(df_constructors)
 
st.header("Constructor Standings Data")
st.dataframe(df_constructor_standings)
 
st.header("Constructor Results Data")
st.dataframe(df_constructor_results)
 
# Formula 1 Constructors Analysis
 
# Filter relevant columns
selected_columns = ['points', 'wins', 'Team_name', 'year', 'name']
df_selected = df_merged_constructors[selected_columns]
 
# Add a filter for selecting the year
selected_year = st.sidebar.selectbox("Select Year", df_selected['year'].unique())
filtered_data = df_selected[df_selected['year'] == selected_year]
 
# Display filtered data
st.header(f"Constructors Data for {selected_year}")
st.dataframe(filtered_data)
 
# Display a heatmap of wins over the years
st.header("Wins of Top 5 Teams Over the Years (Heatmap)")
 
# Identify the top 5 teams with the highest wins over the years
top_teams = df_merged_constructors.groupby('Team_name')['wins'].max().nlargest(5).index
 
# Filter the DataFrame to include only the top 5 teams
top_teams_data = df_merged_constructors[df_merged_constructors['Team_name'].isin(top_teams)]
 
# Create a pivot table
pivot_table = pd.pivot_table(top_teams_data, values='wins', index=['Team_name'], columns=['year'], aggfunc='max')
 
# Create a heatmap with custom colors
plt.figure(figsize=(12, 6))
cbar_kws = {'label': 'Wins'}
sns.heatmap(pivot_table, cmap="coolwarm", annot=True,
            fmt='g', linewidths=.5, cbar_kws=cbar_kws, xticklabels=pivot_table.columns, yticklabels=pivot_table.index,
            annot_kws={"size": 10})
 
st.pyplot(plt)
 
 
# Filter relevant columns
selected_columns = ['points', 'wins', 'Team_name', 'year', 'name']
df_selected = df_merged_constructors[selected_columns]
 
# Add a filter for selecting the teams
selected_teams = st.sidebar.multiselect("Select Teams", df_selected['Team_name'].unique(), default=['Red Bull', 'Mercedes'])
 
# Filter the DataFrame to include only the selected teams
filtered_teams_data = df_selected[df_selected['Team_name'].isin(selected_teams)]
 
# Display selected and filtered data
st.header("Selected and Filtered Constructors Data")
st.dataframe(filtered_teams_data)
 
# Display additional information (e.g., top teams with maximum points each year)
st.header("Top Teams with Maximum Points Each Year")
 
# Create a pivot table
pivot_table_max_points = pd.pivot_table(df_selected, values='points', index=['Team_name'], columns=['year'], aggfunc='max')
 
# Display the top teams with maximum points each year
st.dataframe(pivot_table_max_points.loc[selected_teams])
 
# Calculate the top teams with maximum points overall
df_max_points = df_selected.loc[df_selected.groupby('year')['points'].idxmax()][selected_columns]
 
# Display the top teams with maximum points overall
st.header("Top Teams with Maximum Points Overall")
st.dataframe(df_max_points)
 
# ... (previous code remains unchanged)
 
# Additional Code for Stacked Area Chart
 
# Identify the top 5 teams with the highest scores over the years
top_teams = df_merged_constructors.groupby('Team_name')['points'].sum().nlargest(5).index
 
# Filter the DataFrame to include only the top 5 teams
top_teams_data = df_merged_constructors[df_merged_constructors['Team_name'].isin(top_teams)]
 
team_colors = {
    'Red Bull': '#1E41FF',   # Blue for Red Bull
    'Ferrari': '#DC0000',    # Red for Ferrari
    'Mercedes': '#000000',   # Black for Mercedes
    'Williams': '#808080',   # Grey for Williams
    'McLaren': '#FF8700'     # Orange for McLaren
}
 
# Create a stacked area chart with custom colors
st.header("Top 5 Teams with Highest Constructor Scores Over the Years (Stacked Area Chart)")
 
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='year', y='points', hue='Team_name', data=top_teams_data, palette=team_colors, ci=None, ax=ax)
plt.title('Top 5 Teams with Highest Constructor Scores Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Points')
plt.legend(title='Team', loc='upper left', bbox_to_anchor=(1, 1))
 
# Display the plot using Streamlit
st.pyplot(fig)
 
 
# Sidebar with team selection
selected_team = st.sidebar.selectbox("Select Team", df_max_points['Team_name'].unique())
 
# Filter data for the selected team
team_data = df_max_points[df_max_points['Team_name'] == selected_team]
 
# Display the selected team's name
st.header(f'Wins Distribution for {selected_team}')
 
# Plot histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(team_data['points'], bins=20, color='skyblue', edgecolor='black')
 
# Set labels and title
plt.title(f'Wins Distribution for {selected_team}')
plt.xlabel('Number of Wins')
plt.ylabel('Frequency')
 
# Display the plot using Streamlit
st.pyplot(fig)
 
 
# Read lap times data
lap_times = pd.read_csv('sample_data/lap_times.csv')
 
# Read pit stops data
pit_stops = pd.read_csv('sample_data/pit_stops.csv')
 
# Assuming df_races, df_constructor_standings, and df_constructors are already loaded
 
# Merge dataframes
merged_pit_stops = pd.merge(pit_stops, df_races, left_on='raceId', right_on='raceId')
merged_pit_stops = pd.merge(merged_pit_stops, df_constructor_standings[['raceId', 'constructorId']], on='raceId', how='left')
merged_pit_stops = pd.merge(merged_pit_stops, df_constructors, on='constructorId', how='left')
 
# Select and rename columns
selected_columns = ['name', 'date', 'duration', 'Team_name', 'circuitId']
column_mapping = {
    'name': 'Race_Name',
    'date': 'Race_Date',
    'duration': 'Pit_Stop_Duration',
    'Team_name': 'Constructor_Team',
    'circuitId': 'circuit_id'
}
merged_pit_stops = merged_pit_stops[selected_columns].rename(columns=column_mapping)
 
# Streamlit app
st.title("Pit Stops Data Explorer")
 
# Display the merged pit stops data
st.header("Merged Pit Stops Data")
st.dataframe(merged_pit_stops)
 
# Sidebar for filtering options
selected_team = st.sidebar.selectbox("Select Team", merged_pit_stops['Constructor_Team'].unique())
filtered_data = merged_pit_stops[merged_pit_stops['Constructor_Team'] == selected_team]
 
# Display filtered data
st.header(f"Pit Stops Data for {selected_team}")
st.dataframe(filtered_data)
 
st.title("Latest Pit Stops Data Explorer")
 
# Sort the DataFrame by Race_Date in descending order
df_merged_pit_stops = merged_pit_stops.sort_values(by='Race_Date', ascending=False)
 
# Find the index of the first occurrence of each circuit in the sorted DataFrame
latest_indices = df_merged_pit_stops.groupby('circuit_id').head(1).index
 
# Extract the rows corresponding to the latest occurrences
latest_pit_stops = df_merged_pit_stops.loc[latest_indices]
 
# Display the result
st.header("Latest Pit Stops Data")
st.dataframe(latest_pit_stops[['Race_Name', 'Race_Date', 'Pit_Stop_Duration', 'Constructor_Team', 'circuit_id']])
 
team_counts = latest_pit_stops['Constructor_Team'].value_counts()
 
# Sort the team counts in descending order
team_counts = team_counts.sort_values(ascending=True)
 
# Streamlit app
st.title("Pit Stops Analysis")
 
# Bar chart for number of fastest pit stops by constructor team
st.header("Number of Fastest Pit Stops by Constructor Team")
st.bar_chart(team_counts)
 
# Histogram for distribution of pit stop durations
st.header("Distribution of Pit Stop Durations")
fig, ax = plt.subplots()
ax.hist(latest_pit_stops['Pit_Stop_Duration'], bins=15, color='lightcoral', edgecolor='black')
ax.set_xlabel('Pit Stop Duration')
ax.set_ylabel('Frequency')
st.pyplot(fig)
 
# Line chart for pit stop duration over time
st.header("Pit Stop Duration Over Time")
df_Pit_stop_Duration_over_Time = latest_pit_stops.sort_values('Race_Date')
df_Pit_stop_Duration_over_Time['Race_Date'] = pd.to_datetime(df_Pit_stop_Duration_over_Time['Race_Date'])
 
st.line_chart(df_Pit_stop_Duration_over_Time.set_index('Race_Date')['Pit_Stop_Duration'])
 
st.title("Pit Stops Duration Analysis Over Time")
 
# Merge with circuits data
df_Pit_stop_Duration_over_Time = pd.merge(df_Pit_stop_Duration_over_Time, df_circuits[['circuitId', 'name']], left_on='circuit_id', right_on='circuitId', how='left')
df_Pit_stop_Duration_over_Time = df_Pit_stop_Duration_over_Time.drop(['circuitId'], axis=1)
 
# Check and convert 'Pit_Stop_Duration' column to numeric
df_Pit_stop_Duration_over_Time['Pit_Stop_Duration'] = pd.to_numeric(df_Pit_stop_Duration_over_Time['Pit_Stop_Duration'], errors='coerce')
 
# Team colors
team_colors = {
    'Red Bull': '#1E41FF',   # Blue for Red Bull
    'Ferrari': '#DC0000',    # Red for Ferrari
    'Mercedes': '#000000',   # Black for Mercedes
    'Williams': '#808080',   # Grey for Williams
    'McLaren': '#FF8700',    # Orange for McLaren
    'HRT': '#FFD700',        # Gold for HRT
    'Lotus F1': '#00FF00',   # Green for Lotus F1
    'Caterham': '#FF69B4',   # Pink for Caterham
    'Haas F1 Team': '#8B4513'  # SaddleBrown for Haas F1 Team
}
 
# Extract unique constructor team names from the DataFrame
unique_teams = df_Pit_stop_Duration_over_Time['Constructor_Team'].unique()
 
# Update team_colors dictionary with unique teams and default color if not present
for team in unique_teams:
    team_colors.setdefault(team, '#808080')  # Default to grey if not specified
 
# Bar plot for Pit Stop Duration by Circuit and Team
st.header("Pit Stop Duration by Circuit and Team")
plt.figure(figsize=(14, 8))
sns.barplot(
    x='name',
    y='Pit_Stop_Duration',
    hue='Constructor_Team',
    data=df_Pit_stop_Duration_over_Time,
    ci=None,
    palette=team_colors
)
plt.title('Pit Stop Duration by Circuit and Team')
plt.xlabel('Circuit Name')
plt.ylabel('Pit Stop Duration')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Constructor Team', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
 
# Display the plot using Streamlit
st.pyplot(plt)
 
st.title("Monte Carlo Simulation for Pit Stop Times")
 
# Data Cleaning
df_Pit_stop_Duration_over_Time_cleaned = df_Pit_stop_Duration_over_Time.dropna()
 
# Simulation Setup
num_simulations = 1000  # adjust as needed
simulated_improvements = []
 
# Monte Carlo Simulation
for team in df_Pit_stop_Duration_over_Time_cleaned['Constructor_Team'].unique():
    team_data = df_Pit_stop_Duration_over_Time_cleaned[df_Pit_stop_Duration_over_Time_cleaned['Constructor_Team'] == team]['Pit_Stop_Duration'].values
    mean_improvement = np.mean(team_data)  # Use your own method to define the improvement distribution
    std_dev_improvement = np.std(team_data)  # Use your own method to define the improvement distribution
    
    # Generate improvements for each row in team_data
    improvements = np.random.normal(loc=mean_improvement, scale=std_dev_improvement, size=len(team_data))
    simulated_improvements.append((team, improvements))
 
# Analysis
results = []
for team, improvements in simulated_improvements:
    team_data = df_Pit_stop_Duration_over_Time_cleaned[df_Pit_stop_Duration_over_Time_cleaned['Constructor_Team'] == team]['Pit_Stop_Duration']
    
    # Add the improvements to the team's pit stop times
    simulated_pit_stop_times = team_data + improvements
    
    results.append((team, np.mean(simulated_pit_stop_times), np.std(simulated_pit_stop_times)))
 
# Visualization
results_df = pd.DataFrame(results, columns=['Constructor_Team', 'Mean_Simulated_Pit_Stop', 'Std_Dev_Simulated_Pit_Stop'])
results_df.sort_values(by='Mean_Simulated_Pit_Stop', inplace=True)
 
# Bar plot for Mean Simulated Pit Stop Time with error bars
st.header("Monte Carlo Simulation Results")
plt.figure(figsize=(10, 6))
plt.barh(results_df['Constructor_Team'], results_df['Mean_Simulated_Pit_Stop'], xerr=results_df['Std_Dev_Simulated_Pit_Stop'], capsize=5)
plt.xlabel('Mean Simulated Pit Stop Time')
plt.title('Monte Carlo Simulation Results')
st.pyplot(plt)
 
merged_laps_races = pd.merge(lap_times, df_races[['raceId', 'name', 'circuitId', 'date']], on='raceId', how='left')
 
# Merge with df_drivers to get driver information
merged_laps_races_drivers = pd.merge(merged_laps_races, df_drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
 
# Streamlit app
st.title("Formula 1 Fastest Laps Analysis")
 
print('hey')

# Custom time parser function
def custom_time_parser(time_str):
    try:
        return pd.to_datetime(time_str, format='%H:%M:%S.%f').time()
    except ValueError:
        return pd.to_datetime(time_str, format='%M:%S.%f').time()
print('hi')
# Apply the custom time parser to the 'time' column
merged_laps_races_drivers['time'] = merged_laps_races_drivers['time'].apply(custom_time_parser)

# Convert 'time' column to timedelta
merged_laps_races_drivers['time'] = pd.to_timedelta(merged_laps_races_drivers['time'].astype(str))
print('hello')
# Find the index of the fastest lap for each raceId
fastest_laps_indices = merged_laps_races_drivers.groupby('raceId')['time'].idxmin()

# Extract the rows corresponding to the fastest laps
fastest_laps = merged_laps_races_drivers.loc[fastest_laps_indices]
print('bye')
# Display the result
st.header("Fastest Laps Information")
st.dataframe(fastest_laps[['raceId', 'driverId', 'lap', 'position', 'time', 'name', 'circuitId', 'date', 'forename', 'surname']])


print('bubye')

# Filter entries for Lewis Hamilton (driverId = 1) and Max Verstappen (driverId = 830)
selected_drivers = fastest_laps[fastest_laps['driverId'].isin([1, 830])]
 
# Extract forename entries for Lewis Hamilton and Max Verstappen
lewis_entry = selected_drivers[selected_drivers['driverId'] == 1]
max_entry = selected_drivers[selected_drivers['driverId'] == 830]
 
# Find the common circuits for Lewis Hamilton and Max Verstappen
common_circuits = set(lewis_entry['circuitId']).intersection(max_entry['circuitId'])
 
# Filter entries for common circuits
common_circuits_entries_lewis = lewis_entry[lewis_entry['circuitId'].isin(common_circuits)]
common_circuits_entries_max = max_entry[max_entry['circuitId'].isin(common_circuits)]
 
# Display entries for common circuits
st.header("Common Circuits Entries")
st.subheader("Lewis Hamilton's entries in common circuits:")
st.dataframe(common_circuits_entries_lewis)
 
st.subheader("Max Verstappen's entries in common circuits:")
st.dataframe(common_circuits_entries_max)
 
st.title("Formula 1 Fastest Laps Analysis")
 
# Display the original DataFrame
st.header("Original Fastest Laps Data")
st.dataframe(fastest_laps)
 
# Circuit IDs to filter
selected_circuit_ids = {3, 4, 70, 7, 6, 9, 10, 11, 75, 13, 39, 77, 18, 20, 21, 22, 24}
 
# Driver IDs to filter
selected_driver_ids = [830, 1]  # Replace with the actual driver IDs for Max Verstappen and Lewis Hamilton
 
# Filter the DataFrame
filtered_df = fastest_laps[fastest_laps['circuitId'].isin(selected_circuit_ids) & fastest_laps['driverId'].isin(selected_driver_ids)]
 
# Group by circuitId and driverId, find the index of the minimum lap time
idx_fastest_laps = filtered_df.groupby(['circuitId', 'driverId'])['milliseconds'].idxmin()
 
# Get only the rows corresponding to the indices of the fastest laps
fastest_laps_df = filtered_df.loc[idx_fastest_laps]
 
# Display the DataFrame with only the fastest laps
st.header("Filtered Fastest Laps Data")
st.dataframe(fastest_laps_df)
 
from scipy.stats import ttest_rel
 
# Assuming fastest_laps is your DataFrame
 
# Streamlit app
st.title("Formula 1 Fastest Laps Analysis")
 
# Display the original DataFrame
st.header("Original Fastest Laps Data")
st.dataframe(fastest_laps)
 
# Circuit IDs to filter
selected_circuit_ids = {3, 4, 70, 7, 6, 9, 10, 11, 75, 13, 39, 77, 18, 20, 21, 22, 24}
 
# Driver IDs to filter
selected_driver_ids = [830, 1]  # Replace with the actual driver IDs for Max Verstappen and Lewis Hamilton
 
# Filter the DataFrame
filtered_df = fastest_laps[fastest_laps['circuitId'].isin(selected_circuit_ids) & fastest_laps['driverId'].isin(selected_driver_ids)]
 
# Group by circuitId and driverId, find the index of the minimum lap time
idx_fastest_laps = filtered_df.groupby(['circuitId', 'driverId'])['milliseconds'].idxmin()
 
# Get only the rows corresponding to the indices of the fastest laps
fastest_laps_df = filtered_df.loc[idx_fastest_laps]
 
# Display the DataFrame with only the fastest laps
st.header("Filtered Fastest Laps Data")
st.dataframe(fastest_laps_df)
 
# Convert lap times to milliseconds
fastest_laps_df['lap_time_ms'] = fastest_laps_df['time'].dt.total_seconds() * 1000
 
# Separate the lap times for Max Verstappen and Lewis Hamilton
hamilton_lap_times = fastest_laps_df[fastest_laps_df['driverId'] == 1]['lap_time_ms'].tolist()
verstappen_lap_times = fastest_laps_df[fastest_laps_df['driverId'] == 830]['lap_time_ms'].tolist()
 
# Perform a paired t-test
t_stat, p_value = ttest_rel(hamilton_lap_times, verstappen_lap_times)
 
# Display the t-test results
st.header("Paired t-test for Fastest Laps")
st.write(f"T-statistic: {t_stat}")
st.write(f"P-value: {p_value}")