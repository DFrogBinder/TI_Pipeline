import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.factorplots import interaction_plot

def Electrode_Size(data):
    # Filter the data where 'Pair 1 Position' equals 'AF4-PO4'
    data = data[data['Pair 1 Position'] == 'AF4-PO4']
    
    # Grouping the data by electrode size for analysis
    groups_volume = data.groupby('Electrode Size')['80_Percent_Cutoff_Volume'].apply(list)
    groups_max_thalamus = data.groupby('Electrode Size')['Max_Thalamus'].apply(list)
    groups_max_value = data.groupby('Electrode Size')['Maximum Value'].apply(list)

    # Performing ANOVA to test the hypothesis for different parameters
    anova_volume = f_oneway(*groups_volume)
    anova_max_thalamus = f_oneway(*groups_max_thalamus)
    anova_max_value = f_oneway(*groups_max_value)

    # Output ANOVA results
    print("ANOVA results for Total Volume:", anova_volume)
    print("ANOVA results for Max Thalamus Intensity:", anova_max_thalamus)
    print("ANOVA results for Max Overall Brain Intensity:", anova_max_value)

    # Determine the order for 'Electrode Size' (assuming they are strings like '1mm', '2mm')
    # Convert to numeric if necessary, or sort based on your specific rules
    electrode_sizes = sorted(data['Electrode Size'].unique(), key=lambda x: float(x.rstrip('cm')))

    # Visualization with boxplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.boxplot(x='Electrode Size', y='Max_Thalamus', palette="muted", data=data, ax=axes[0], order=electrode_sizes)
    axes[0].set_title('Peak E-field (Thalamus) by Electrode Size')
    axes[0].set_ylabel('Peak E-field (V/m)')

    sns.boxplot(x='Electrode Size', y='Maximum Value', palette="muted", data=data, ax=axes[1], order=electrode_sizes)
    axes[1].set_title('Peak E-field (Overall) Brain by Electrode Size')
    axes[1].set_ylabel('Peak E-field (V/m)')

    plt.tight_layout()
    plt.show()

    return anova_volume, anova_max_thalamus, anova_max_value

def Electrode_Shape_Size(data):
    data.columns = [col.replace(' ', '_') for col in data.columns]

    # ANOVA for Max_Thalamus
    model_thalamus = ols('Max_Thalamus ~ C(Electrode_Size) * C(Electrode_Shape)', data=data).fit()
    anova_thalamus = sm.stats.anova_lm(model_thalamus, typ=2)

    # ANOVA for Maximum Value in overall brain
    model_max_value = ols('Maximum_Value ~ C(Electrode_Size) * C(Electrode_Shape)', data=data).fit()
    anova_max_value = sm.stats.anova_lm(model_max_value, typ=2)

    # ANOVA for Total Stimulation Volume
    model_total_volume = ols('Total_Volume ~ C(Electrode_Size) * C(Electrode_Shape)', data=data).fit()
    anova_total_volume = sm.stats.anova_lm(model_total_volume, typ=2)

    # Creating a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10,12))  # 3 rows, 1 column

    # Interaction plot for Max_Thalamus
    interaction_plot(data['Electrode_Size'], data['Electrode_Shape'], data['Max_Thalamus'],
                    ax=axs[0], colors=['red', 'blue'], markers=['D', '^'], ms=10)
    axs[0].set_title('Peak E-field Intesity in Thalamus')
    axs[0].set_xlabel('Electrode Size')
    axs[0].set_ylabel('Peak E-field Intesity (V/m)')
    axs[0].legend(title='Electrode Shape')

    # Interaction plot for Maximum Value in the overall brain
    interaction_plot(data['Electrode_Size'], data['Electrode_Shape'], data['Maximum_Value'],
                    ax=axs[1], colors=['red', 'blue'], markers=['D', '^'], ms=10)
    axs[1].set_title('Peak E-field Intesity  in Overall Brain')
    axs[1].set_xlabel('Electrode Size')
    axs[1].set_ylabel('Peak E-field Intesity ')
    axs[1].legend(title='Electrode Shape')

    # Interaction plot for Total Stimulation Volume
    interaction_plot(data['Electrode_Size'], data['Electrode_Shape'], data['Total_Volume'],
                    ax=axs[2], colors=['red', 'blue'], markers=['D', '^'], ms=10)
    axs[2].set_title('Total Stimulation Volume')
    axs[2].set_xlabel('Electrode Size')
    axs[2].set_ylabel('Volume')
    axs[2].legend(title='Electrode Shape')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Increase the vertical space between the plots

    plt.show()
    
    
def Electrode_Shape(data):
    
    # Filter the data where 'Pair 1 Position' equals 'AF4-PO4'
    data = data[data['Pair 1 Position'] == 'AF4-PO4']
    
    # Set up the visualizations
    sns.set(style="whitegrid")

    # Set font sizes for all figures via rcParams
    plt.rcParams['axes.labelsize'] = 28  # Sets the default axes labels size
    plt.rcParams['xtick.labelsize'] = 20  # Sets the x-axis tick labels size
    plt.rcParams['ytick.labelsize'] = 20  # Sets the y-axis tick labels size
    plt.rcParams['axes.titlesize'] = 30  # Sets the default title size
    
    # Plotting the distribution of maximum intensity in the thalamus for each electrode shape
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Electrode Shape', y='Max_Thalamus', data=data, palette="muted")
    plt.title('Distribution of Maximum Intensity in the Thalamus by Electrode Shape')
    plt.xlabel('Electrode Shape')
    plt.ylabel('Peak E-field Intesity (V/m)')
    plt.show()

    # Extracting the groups based on electrode shape
    ellipse_data = data[data['Electrode Shape'] == 'ellipse']['Max_Thalamus']
    rect_data = data[data['Electrode Shape'] == 'rect']['Max_Thalamus']

    # Performing an independent t-test
    t_stat, p_value = ttest_ind(ellipse_data, rect_data, equal_var=False)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)
    
    return t_stat, p_value    
    
def perform_linear_regression(x, y):
        model = LinearRegression()
        x_values = x.values.reshape(-1, 1)  # Reshape for sklearn
        y_values = y.values
        model.fit(x_values, y_values)
        y_pred = model.predict(x_values)
        return x_values, y_pred, model.coef_[0], model.intercept_

def Max_vs_Current(data):
     # Convert Input Current from string to numeric (e.g., "2mA" to 2)
    data['Input Current Numeric'] = data['Input Current'].str.extract('(\d+\.?\d*)').astype(float)
    
    # Generate 'Pair Position' column from 'pair 1 position' and 'pair 2 position'
    data['Pair Position'] = data['Pair 1 Position'].astype(str) + "-" + data['Pair 2 Position'].astype(str)

    # Getting unique pair positions and setting up a color palette
    unique_pairs = data['Pair Position'].unique()
    palette = sns.color_palette("hsv", len(unique_pairs))
    pair_color_mapping = dict(zip(unique_pairs, palette))
    
    # Create correlation matrix for 'Input Current Numeric', 'Maximum Value', and 'Max_Thalamus'
    correlation_matrix = data[['Input Current Numeric', 'Maximum Value', 'Max_Thalamus']].corr()

    # Function to perform linear regression
    def perform_linear_regression(x, y):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        x = x.values.reshape(-1, 1)  # Reshaping for sklearn
        model.fit(x, y)
        y_pred = model.predict(x)
        return x, y_pred, model.coef_[0], model.intercept_

    # Data for regression
    x_values_max_value, y_pred_max_value, coef_max_value, intercept_max_value = perform_linear_regression(data['Input Current Numeric'], data['Maximum Value'])
    x_values_max_thalamus, y_pred_max_thalamus, coef_max_thalamus, intercept_max_thalamus = perform_linear_regression(data['Input Current Numeric'], data['Max_Thalamus'])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plots colored by 'Pair Position'
    for pair in unique_pairs:
        subset = data[data['Pair Position'] == pair]
        sns.scatterplot(x=subset['Input Current Numeric'], y=subset['Maximum Value'], ax=ax[0], label=f'Pair {pair}', color=pair_color_mapping[pair])
    ax[0].plot(x_values_max_value.flatten(), y_pred_max_value, color='red', label=f'Linear Fit: y={coef_max_value:.2f}x+{intercept_max_value:.2f}')
    ax[0].set_title('Max. Intensity (Brain) vs. Input Current')
    ax[0].set_xlabel('Input Current (mA)')
    ax[0].set_ylabel('Peak E-field Intesity (V/m)')
    ax[0].legend(fontsize='large') 

    for pair in unique_pairs:
        subset = data[data['Pair Position'] == pair]
        sns.scatterplot(x=subset['Input Current Numeric'], y=subset['Max_Thalamus'], ax=ax[1], label=f'Pair {pair}', color=pair_color_mapping[pair])
    ax[1].plot(x_values_max_thalamus.flatten(), y_pred_max_thalamus, color='red', label=f'Linear Fit: y={coef_max_thalamus:.2f}x+{intercept_max_thalamus:.2f}')
    ax[1].set_title('Max. Intensity (Thalamus) vs. Input Current')
    ax[1].set_xlabel('Input Current (mA)')
    ax[1].set_ylabel('Peak E-field Intesity (V/m)')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
    # Print correlation matrix
    print(correlation_matrix)
    return correlation_matrix

def ElectrodePossition(data):
    # Combine the positions into a single column for interaction effects
    data['Electrode_Positions'] = data['Pair 1 Position'] + ' / ' + data['Pair 2 Position']

    # Perform ANOVA to test the effect of electrode positions on Max Thalamus
    anova_model = ols('Max_Thalamus ~ Electrode_Positions', data=data).fit()
    anova_results = sm.stats.anova_lm(anova_model, typ=2)  # Type 2 ANOVA DataFrame

    # Print ANOVA results
    print(anova_results)

    # Prepare data for visualization
    grouped_positions = data.groupby('Electrode_Positions')['Max_Thalamus'].mean().sort_values(ascending=False)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    grouped_positions.plot(kind='bar')
    plt.title('Average Max. Thalamus Intensity by Electrode Positions')
    plt.xlabel('Electrode Positions')
    plt.ylabel('Average Max Intensty')
    plt.xticks(rotation=45, ha='right')  # Slanting the x-axis text for better readability
    plt.tight_layout()
    plt.show()
        
    # Create the boxplot with adjusted aesthetics
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Electrode_Positions', y='Max_Thalamus', data=data,
                linewidth=2.5,  # Makes the outlines of the boxes thicker
                palette="muted")  # Uses a matte color palette
    plt.title('Distribution of Max Thalamus Intensity Across Electrode Positions')
    plt.xlabel('Electrode Positions')
    plt.ylabel('Peak E-field Intesity (V/m)')
    plt.xticks(rotation=45, ha='right')  # Slanting the x-axis text for better readability
    plt.tight_layout()
    plt.show()
    
    return anova_results

# Load the CSV file
data = pd.read_csv('/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/All_Stats.csv')

# Set font sizes for all figures via rcParams
plt.rcParams['axes.labelsize'] = 28  # Sets the default axes labels size
plt.rcParams['xtick.labelsize'] = 14  # Sets the x-axis tick labels size
plt.rcParams['ytick.labelsize'] = 14  # Sets the y-axis tick labels size
plt.rcParams['axes.titlesize'] = 29  # Sets the default title size


# cm = Max_vs_Current(data)
# anova_stats = ElectrodePossition(data)
# t,p = Electrode_Shape(data)
Electrode_Size(data)