from openpyxl import load_workbook

# Load the workbook and access the "Plots" sheet
wb = load_workbook('CamCAN.xlsx', data_only=False)
ws_plots = wb['Plots']

# Check if there are any chart objects in the "Plots" sheet
if hasattr(ws_plots, '_charts'):
    charts = ws_plots._charts
    print(f"Found {len(charts)} chart(s) in the 'Plots' sheet.")
    for idx, chart in enumerate(charts, start=1):
        print(f"\nChart {idx}:")
        try:
            print("Title:", chart.title)
        except Exception as e:
            print("Title: (not set)", e)
        for s in chart.series:
            # Try to print the 'values' attribute if available
            try:
                print("Series data range (values):", s.values)
            except AttributeError:
                # For XYSeries objects, use xvalues and yvalues
                print("This series appears to be an XYSeries.")
                x_values = getattr(s, 'xvalues', None)
                y_values = getattr(s, 'yvalues', None)
                print("X values:", x_values)
                print("Y values:", y_values)
                # Optionally, inspect all available attributes of the series:
                # print("Available attributes:", dir(s))
else:
    print("No charts found in the 'Plots' sheet.")
