import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.factorplots import interaction_plot

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
    axs[0].set_title('Max Intensity in Thalamus')
    axs[0].set_xlabel('Electrode Size')
    axs[0].set_ylabel('Max Intensity')
    axs[0].legend(title='Electrode Shape')

    # Interaction plot for Maximum Value in the overall brain
    interaction_plot(data['Electrode_Size'], data['Electrode_Shape'], data['Maximum_Value'],
                    ax=axs[1], colors=['red', 'blue'], markers=['D', '^'], ms=10)
    axs[1].set_title('Max Intensity in Overall Brain')
    axs[1].set_xlabel('Electrode Size')
    axs[1].set_ylabel('Max Intensity')
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
    # Set up the visualizations
    sns.set(style="whitegrid")

    # Plotting the distribution of maximum intensity in the thalamus for each electrode shape
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Electrode Shape', y='Max_Thalamus', data=data, palette="muted")
    plt.title('Distribution of Maximum Intensity in the Thalamus by Electrode Shape')
    plt.xlabel('Electrode Shape')
    plt.ylabel('Maximum Intensity')
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

    # Create correlation matrix for 'Input Current Numeric', 'Maximum Value', and 'Max_Thalamus'
    correlation_matrix = data[['Input Current Numeric', 'Maximum Value', 'Max_Thalamus']].corr()

    # Function to perform linear regression and return necessary values for plotting
    

    # Data for regression
    x_values_max_value, y_pred_max_value, coef_max_value, intercept_max_value = perform_linear_regression(data['Input Current Numeric'], data['Maximum Value'])
    x_values_max_thalamus, y_pred_max_thalamus, coef_max_thalamus, intercept_max_thalamus = perform_linear_regression(data['Input Current Numeric'], data['Max_Thalamus'])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=data['Input Current Numeric'], y=data['Maximum Value'], ax=ax[0], color='blue', label='Actual Data')
    ax[0].plot(x_values_max_value.flatten(), y_pred_max_value, color='red', label=f'Linear Fit: y={coef_max_value:.2f}x+{intercept_max_value:.2f}')
    ax[0].set_title('Max. Intensity (Brain) vs. Input Current')
    ax[0].set_xlabel('Input Current (mA)')
    ax[0].set_ylabel('Maximum Intensity')
    ax[0].legend()

    sns.scatterplot(x=data['Input Current Numeric'], y=data['Max_Thalamus'], ax=ax[1], color='green', label='Actual Data')
    ax[1].plot(x_values_max_thalamus.flatten(), y_pred_max_thalamus, color='red', label=f'Linear Fit: y={coef_max_thalamus:.2f}x+{intercept_max_thalamus:.2f}')
    ax[1].set_title('Max. Intensity (Thalamus) vs. Input Current')
    ax[1].set_xlabel('Input Current (mA)')
    ax[1].set_ylabel('Max. Intensity')
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
    plt.ylabel('Max Intensity')
    plt.xticks(rotation=45, ha='right')  # Slanting the x-axis text for better readability
    plt.tight_layout()
    plt.show()
    
    return anova_results

# Load the CSV file
data = pd.read_csv('/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/All_Stats.csv')
# cm = Max_vs_Current(data)
# anova_stats = ElectrodePossition(data)
# t,p = Electrode_Shape(data)
Electrode_Shape_Size(data)