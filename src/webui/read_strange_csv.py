import pandas as pd


def read_and_save_csv(input_file_path, output_file_path):
    """
    Reads a CSV file, treating each line as a single data entry, and saves it to a new file.

    Parameters:
    - input_file_path: str, the path to the input CSV file.
    - output_file_path: str, the path where the reformatted CSV file will be saved.
    """
    # Initialize a list to hold the data
    data = []

    # Read the file line by line
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Append each line as a single entry in the list
            data.append([line.strip()])

    # Create a DataFrame from the list
    df = pd.DataFrame(data, columns=['Text'])

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f'File saved successfully to {output_file_path}')

if __name__ == "__main__":
    input = "/home/jeff/Downloads/data3.csv"
    # read the file
    try:
        df = pd.read_csv(input)
        # print the first 5 rows of the dataframe

        print(df.head())
        # pandas.errors.ParserError: Error tokenizing data. C error: Expected 6 fields in line 3, saw 11
    except pd.errors.ParserError as e:
        print(e)
        print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed.")
        # fix the file
        output = "/home/jeff/Downloads/data3_fixed.csv"
        read_and_save_csv(input, output)
        # read the fixed file
        df = pd.read_csv(output)
        # print the first 5 rows of the dataframe
        print(df.head())


