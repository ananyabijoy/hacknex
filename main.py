import pandas as pd
import os

def process_file(file_path):

    # Read CSV or Excel file
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)

    else:
        print("Unsupported file type")
        return

    print("\n========== DATA PROCESSING ==========")

    # Number of rows and columns
    print("\nRows:", data.shape[0])
    print("Columns:", data.shape[1])

    # Column names
    print("\nCOLUMN NAMES:")
    print(list(data.columns))

    # Data types
    print("\nDATA TYPES:")
    print(data.dtypes)

    # Missing values
    print("\nMISSING VALUES:")
    print(data.isnull().sum())

    # Duplicate rows
    print("\nDUPLICATE ROWS:")
    print(data.duplicated().sum())

    # Display data
    print("\nDATA:")
    print(data.to_string(index=False))

    print("\n========== PROCESSING COMPLETE ==========")


# Find CSV and Excel files
files = []

for file in os.listdir("."):
    if file.endswith(".csv") or file.endswith(".xlsx"):
        files.append(file)

if len(files) == 0:
    print("No CSV or Excel files found.")

else:
    print("Available files:")

    for i, file in enumerate(files, 1):
        print(i, "-", file)

    choice = int(input("\nEnter file number: "))

    selected_file = files[choice - 1]

    print("\nProcessing:", selected_file)

    process_file(selected_file)