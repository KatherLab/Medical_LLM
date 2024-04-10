import json
import re

import pandas as pd


def postprocess(file_path, pattern, variable, default_pos, default, csv_file_path):
    reports = []
    variables = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            report = data.get('report', '')
            assistant_response = data.get('content', '')

            primary_matches = re.findall(pattern, assistant_response, re.IGNORECASE)

            if not primary_matches:
                matched_text = default
            else:
                matched_text = ', '.join(primary_matches)

            reports.append(report)
            variables.append(matched_text)

    df = pd.DataFrame({
        'report': reports,
        f'{variable}': variables,
    })

    df.to_csv(f'{csv_file_path}')
