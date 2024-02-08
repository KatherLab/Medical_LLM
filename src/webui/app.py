import os
import sqlite3
import subprocess
from typing import Any, Optional
from collections.abc import Iterable
from pathlib import Path
import json

import pandas as pd
import paramiko
from flask import Flask, jsonify, request, send_file, render_template
from flask import current_app
from flask import redirect, url_for
import requests

# import file_to_json function
from input_parser import file_to_json

app = Flask(__name__)

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None


app.config['DOWNLOAD_FOLDER'] = '/home/jeff/PycharmProjects/Medical_LLM/temp_output'
app.config['DB_PATH'] = '/home/jeff/LLM_database/mimicIV.db'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/go', methods=['POST'])
def go():
    # check if the post request has the file part
    if 'file' not in request.files:
        return "No file part in the request.", 400

    file = request.files['file']

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return "No selected file.", 400
    
    model_dir = Path("/mnt/bulk/isabella/llamaproj")
    
    model_path = model_dir/request.form["model"]
    assert model_path.absolute().parent == model_dir

    global server_connection, current_model
    if current_model != request.form["model"]:
        server_connection and server_connection.kill()
        server_connection = subprocess.Popen([
            "/mnt/bulk/isabella/llamaproj/llama.cpp/server",
            "--model", str(model_path),
            "--ctx-size", "2048",
            "--n-gpu-layers", "100",
            "--verbose"
        ])
        current_model = request.form["model"]
    
    df = pd.read_csv(file)
    variables = [var.strip() for var in request.form["variables"].split(",")]
    results = extract_from_report(df, prompt=request.form["prompt"], symptoms=variables, temperature=float(request.form["temperature"]))
    breakpoint()
    

def extract_from_report(df: pd.DataFrame, prompt: str, symptoms: Iterable[str], temperature: float) -> dict[Any]:
    results = {}
    for report in df.report:
        for symptom in symptoms:
            while True:
                try:
                    result = requests.post(
                        url="http://localhost:8080/completion",
                        json={
                            "prompt": prompt.format(symptom=symptom, report="".join(report)),
                            "n_predict": 2048,
                            "temperature": temperature,
                        },
                    )
                    summary = result.json()
                    break
                except json.decoder.JSONDecodeError:
                    pass
            if report not in results:
                results[report] = {}
            results[report][symptom] = summary
            breakpoint()

    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Call the file_to_json function and store the result
        transformed_text = file_to_json(filepath)

        # Retrieve other form data
        condition = request.form.get('condition')
        icd_code = request.form.get('icd_code')

        if not condition or not icd_code:  # Simple check for required fields
            return "Error: Missing required fields", 400

        gender = request.form.get('gender')
        birthdate = request.form.get('birthdate')
        model_selection = request.form.get('model_selection')
        temperature = request.form.get('temperature')
        disease_type = request.form.get('disease_type')
        symptom = request.form.get('symptom')
        language = request.form.get('language')
        export_format = request.form.get('export_format')

        # Process the uploaded CSV file
        df = pd.read_csv(filepath)
        # You can add your data processing logic here


        try:
            # Data processing logic based on user input
            processed_data = analyze_data(df, condition, icd_code, gender, birthdate, model_selection, temperature,
                                          disease_type, symptom, language, export_format)
            return render_template('results.html', data=processed_data)
        except Exception as e:
            # Handle exceptions
            return f"An error occurred: {str(e)}", 500


        # Redirect to a results page or display results
        return redirect(url_for('results'))

    return redirect(url_for('index'))


from postprecessing import postprocess


@app.route('/postprocess', methods=['POST'])
def postprocess_route():
    file_path = request.form.get('file_path')
    pattern = request.form.get('pattern')
    variable = request.form.get('variable')
    default_pos = request.form.get('default_pos')
    default = request.form.get('default')
    csv_file_path = request.form.get('csv_file_path')

    postprocess(file_path, pattern, variable, default_pos, default, csv_file_path)

    return "Postprocessing completed.", 200


@app.route('/extract', methods=['POST'])
def extract():
    output_json = request.form.get('output_json')
    prompt = request.form.get('prompt')
    symptoms = request.form.get('symptoms').split(',')
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extract_from_report(output_json, prompt, symptoms, filepath)

        return "Extraction completed.", 200

    return "Invalid file type.", 400


@app.route('/llm_server_config', methods=['POST'])
def llm_server_config():
    # Retrieve server configuration from form data
    ip_address = request.form.get('ip_address')
    user_name = request.form.get('user_name')
    keyfile_path = request.form.get('keyfile_path')
    model = request.form.get('model_selection')

    # Validate the inputs
    if not ip_address or not user_name or not keyfile_path:
        return "Error: Missing required fields", 400
    
    # Establish a connection to the server
    global server_connection
    if server_connection is not None:
        server_connection.kill()
    server_connection = subprocess.Popen([
        "ssh", "-tt",
        "-i", keyfile_path,
        "isabella@192.168.33.114",
        "/mnt/bulk/isabella/llamaproj/llama.cpp/server",
        "--model", model,
        "--ctx-size", "2048",
        "--n-gpu-layers", "100",
        "--verbose"
    ])

    # server_connection = ServerConnectionClass(ip_address, user_name, password)
    
    # try:
    #     server_connection.connect()
    # except Exception as e:
    #     return f"An error occurred while connecting to the server: {str(e)}", 500

    # If the connection is successful, redirect to a new page or return a success message
    return "Successfully connected to the server.", 200

@app.route('/results')
def results():
    # Logic to display results
    return render_template('results.html')

# Additional endpoint for the database query interface
@app.route('/db_query')
def db_query():
    return render_template('db_query.html')

def get_db_connection():
    conn = sqlite3.connect(current_app.config['DB_PATH'])
    conn.row_factory = sqlite3.Row
    return conn

# Endpoint for getting unique values for dropdowns
@app.route('/get_unique_values')
def get_unique_values():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch unique values from specified columns
    cursor.execute("SELECT DISTINCT race FROM ed_edstays;")
    races = cursor.fetchall()

    cursor.execute("SELECT DISTINCT name FROM ed_medrecon;")
    medrecon_names = cursor.fetchall()

    cursor.execute("SELECT DISTINCT name FROM ed_pyxis;")
    pyxis_names = cursor.fetchall()

    conn.close()

    # Convert query results to lists
    races = [x['race'] for x in races]
    medrecon_names = [x['name'] for x in medrecon_names]
    pyxis_names = [x['name'] for x in pyxis_names]

    return jsonify({'races': races, 'medrecon_names': medrecon_names, 'pyxis_names': pyxis_names})

# Endpoint for querying unique subject_ids based on selected criteria
@app.route('/query_subject_ids', methods=['POST'])
def query_subject_ids():
    race = request.json.get('race')
    medrecon_name = request.json.get('medrecon_name')
    pyxis_name = request.json.get('pyxis_name')

    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT DISTINCT nd.subject_id FROM note_discharge nd "
    conditions = []

    if race:
        conditions.append(f"ees.race = '{race}'")
    if medrecon_name:
        conditions.append(f"emr.name = '{medrecon_name}'")
    if pyxis_name:
        conditions.append(f"et.name = '{pyxis_name}'")

    if conditions:
        query += "JOIN ed_edstays ees ON nd.subject_id = ees.subject_id "
        query += "JOIN ed_medrecon emr ON nd.subject_id = emr.subject_id "
        query += "JOIN ed_pyxis et ON nd.subject_id = et.subject_id "
        query += "WHERE " + " AND ".join(conditions)

    cursor.execute(query)
    subject_ids = cursor.fetchall()
    conn.close()

    return jsonify([x['subject_id'] for x in subject_ids])



# Endpoint for downloading note_discharge text for a subject_id
@app.route('/download_text/<subject_id>')
def download_text(subject_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT text FROM note_discharge WHERE subject_id = ?", (subject_id,))
    text = cursor.fetchone()
    conn.close()

    if text:
        temp_dir = current_app.config['DOWNLOAD_FOLDER']
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)  # Create the temp directory if it doesn't exist

        file_path = os.path.join(temp_dir, f"temp_{subject_id}.txt")
        try:
            with open(file_path, 'w') as file:
                file.write(text['text'])
            return send_file(file_path, as_attachment=True, download_name=f"note_discharge_{subject_id}.txt")
        except IOError as e:
            print(f"Error writing file: {e}")
            return "Error creating the file", 500
    else:
        return "No text found for the provided subject_id", 404


class ServerConnectionClass:
    def __init__(self, ip_address, username, password):
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect(self):
        try:
            self.ssh.connect(self.ip_address, username=self.username, password=self.password)
        except Exception as e:
            raise Exception(f"An error occurred while connecting to the server: {str(e)}")

    def execute_command(self, command):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        return stdout.read()

    def close(self):
        self.ssh.close()

if __name__ == '__main__':
    app.run(debug=True)
