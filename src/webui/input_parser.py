import json
import logging
import os

import PyPDF2
import pandas as pd


# Set up logging
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def file_to_json(dir_path):
    try:
        # Parse the directory path to get the file name and extension
        file_name, file_extension = os.path.splitext(dir_path)

        # Initialize an empty dictionary to store the file content
        file_content = {}

        # Check the file extension and read the file accordingly
        if file_extension == '.txt':
            with open(dir_path, 'r') as file:
                file_content[file_name] = file.read()
        elif file_extension == '.csv':
            df = pd.read_csv(dir_path)
            file_content[file_name] = df.to_dict()
        elif file_extension == '.pdf':
            pdf_file_obj = open(dir_path, 'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
            text = ''
            for page_num in range(pdf_reader.numPages):
                page_obj = pdf_reader.getPage(page_num)
                text += page_obj.extractText()
            file_content[file_name] = text
            pdf_file_obj.close()
        else:
            return "Unsupported file format"

        # Convert the file content to JSON format and return
        return json.dumps(file_content)

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return str(e)
