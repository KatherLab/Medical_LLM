import io
import re
import os
import secrets
import subprocess
import time
from collections.abc import Iterable
from concurrent import futures
from pathlib import Path
from typing import Any, Optional
import pdfplumber
from read_strange_csv import read_and_save_csv
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, send_file, Response, url_for, flash
from markupsafe import Markup
from flask_socketio import SocketIO
from argparse import ArgumentParser
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, BooleanField, TextAreaField, MultipleFileField, FileField, FloatField, validators, SelectField
from wtforms.validators import DataRequired, ValidationError
from enum import Enum
import ast

import re

app = Flask(__name__)

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None

JobID = str
jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

app.config["DOWNLOAD_FOLDER"] = "/home/jeff/PycharmProjects/Medical_LLM/temp_output"
app.config["DB_PATH"] = "/home/jeff/LLM_database/mimicIV.db"
app.secret_key = '789w4hriodfslu9thwioisap89p34hrbwei9pzr8w9e'

socketio = SocketIO(app)

parser = ArgumentParser(description="Parameters to run the KatherLab LLM Pipeline")
parser.add_argument("--model_path", type=str, default=r"D:\LLM-Pipeline\models", help="Path where the models are stored which llama cpp can load.")
parser.add_argument("--server_path", type=str, default=r"D:\LLM-Pipeline\llama-b2453-bin-win-cublas-cu12.2.0-x64/server.exe")
parser.add_argument("--port", type=int, default=5001, help="On which port the Web App should be available.")
parser.add_argument("--ctx_size", type=int, default=2048)
parser.add_argument("--n_gpu_layers", type=int, default=100)
parser.add_argument("--n_predict", type=int, default=2048)

args = parser.parse_args()

class FileMergeForm(FlaskForm):
    file = MultipleFileField("File", validators=[
        FileRequired(),  
        FileAllowed(['txt', 'xlms'], 'Only .txt or .xlms files allowed!')
    ])
    submit = SubmitField("Merge")

default_prompt = r"""[INST] &lt;&lt;SYS&gt;&gt;
Sie sind ein medizinischer Assistent von OpenAI ChatGPT. Im Folgenden finden Sie einen Bericht einer neuroradiologischen Intervention. Manchmal sind nicht alle Informationen angegeben. Wenn Informationen fehlen, geben Sie "null" an. Bitte extrahieren Sie die gesuchten Informationen aus dem Interventionsbericht. 
&lt;&lt;/SYS&gt;&gt;
[/INST]
Beispiel: 
Untersuchung vom 22.12.2022.

Indikation: Notfallintervention ohne vorherige schriftliche Einverständniserklärung bei rechtshemisphärieller Schlaganfallsymptomatik mit CT-angiographisch nachgewiesenem prox. M2-Verschluss bei wohl thrombosiertem Aneurysma, ASPECT = 10. Thrombektomie frustran. IV Lyse erfolgt. NihSS: 5. 

Kam es zu einer Blutung? Ja, wenn eine Blutung oder SAB beschrieben wird. Nein, wenn im Befund steht, dass keine Blutung vorliegt. 
ASSISTANT: nein

[INST]
Das ist der Interventionsbericht:
{report}

Kam es zu einer {symptom}? 
[/INST]"""

class FileExistsValidator:
    def __init__(self, message=None, path=""):
        self.message = message or 'File does not exist.'
        self.path = path

    def __call__(self, form, field):
        filename = os.path.join(self.path, field.data)
        if not os.path.exists(filename):
            raise ValidationError(self.message)

class GrammarValidator:
    def __call__(self, form, field):
        enable_grammar = form.enable_grammar.data
        grammar = field.data#
        if enable_grammar:
            print("Check grammar")
        if enable_grammar and not grammar:
            raise ValidationError('Grammar field is required when "Enable Grammar" is checked.')

class LLMPipelineForm(FlaskForm):
    file = FileField("File", validators=[
        FileRequired(),  
        FileAllowed(['csv', 'xlms'], 'Only .csv and .xlms files allowed!')
    ])
    enable_grammar = BooleanField("Enable Grammar", validators=[])
    grammar = TextAreaField("Grammar:", validators=[GrammarValidator()])
    prompt = TextAreaField("Prompt:", validators=[], default=default_prompt)
    variables = StringField("Variables (separated by commas):", validators=[], default="Headache,Cancer")
    temperature = FloatField("Temperature:", validators=[validators.NumberRange(0,1)], default=0.7)
    model = SelectField("Model:", choices=[("sauerkrautlm-7b-v1.Q5_K_M.gguf", "LLaMA 2 7b chat sauerkraut"), ("sauerkrautlm-70b-v1.Q5_K_M.gguf", "LLaMA 2 70b chat sauerkraut"), ("sauerkrautlm-13b-v1.Q5_K_M.gguf", "LLaMA 2 13b chat sauerkraut")], validators=[FileExistsValidator(message='File does not exist.', path=args.model_path)])

    pattern = StringField("Pattern:", validators=[], default="ja|nein")
    default_grammar = StringField("Default Grammar:", validators=[], default="fehlt")

    submit = SubmitField("Run Pipeline")

class PostProcessType(Enum):
    DEFAULT = 1
    GRAMMAR = 2



@app.route("/", methods=["GET", "POST"])
def index():

    # if "submit-mergeform" in request.form:
    #     mergeform = FileMergeForm(request.form)
    #     form = LLMPipelineForm()
    # elif "submit-form" in request.form:
    #     mergeform = FileMergeForm()
    #     form = LLMPipelineForm(request.form)
    # else:
    mergeform = FileMergeForm()
    form = LLMPipelineForm()

    if "submit-mergeform" in request.form and mergeform.validate_on_submit():
        merged_data = []

        files = request.files.getlist("file")
        for file in files:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                merged_data.append(df)
            elif file.filename.endswith('.pdf'):
                text = ''
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text()
                merged_data.append(pd.DataFrame({'report': [text]}))
            elif file.filename.endswith('.txt'):
                text = file.stream.read().decode('utf-8')
                merged_data.append(pd.DataFrame({'report': [text]}))

        merged_df = pd.concat(merged_data)
        merged_csv = merged_df.to_csv(index=False)
        return Response(
            merged_csv,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=merged.csv"},
        )

        return render_template("index.html", form=form, mergeform=mergeform)    

    if "submit-form" in request.form and form.validate_on_submit():
        file = request.files["file"]

        if file.filename.endswith('.csv'):
            try:
                print(file)
                df = pd.read_csv(file)
            except pd.errors.ParserError as e:
                # print the error message in console
                print(e)
                print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed. Usually, this is caused by a line break in a field. The file will be fixed and then read again.")
                # fix the file
                fixed_file = io.BytesIO()
                read_and_save_csv(file, fixed_file)
                fixed_file.seek(0)
                df = pd.read_csv(fixed_file)
        
        elif file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file)
                print(df.head())
                # ValueError: Excel file format cannot be determined, you must specify an engine manually.
            except ValueError as e:
                print(e)
                print("The error message indicates that the Excel file format cannot be determined. This means that the Excel file is not properly formatted and needs to be fixed. The file will be fixed and then read again.")
                # fix the file
                flash("Excel file is not properly formatted!", "danger")
                return render_template("index.html", form=form, mergeform=mergeform)

        variables = [var.strip() for var in form.variables.data.split(",")]
        job_id = secrets.token_urlsafe()

        if not os.path.exists(args.server_path):
            flash("Llama CPP Server executable not found. Did you specify --server_path correctly?", "danger")
            return render_template("index.html", form=form, mergeform=mergeform)

        if form.enable_grammar.data:
            postprocess_type = PostProcessType.GRAMMAR
        else:
            postprocess_type = PostProcessType.DEFAULT

        global jobs
        jobs[job_id] = executor.submit(
            extract_from_report,
            df=df,
            model_name=form.model.data,
            prompt=form.prompt.data,
            symptoms=variables,
            temperature=float(form.temperature.data),
            pattern=form.pattern.data,
            default=form.default_grammar.data,
            grammar=form.grammar.data.replace("\r\n", "\n"),
            postprocess_type=postprocess_type
        )

        return redirect(url_for('result', job=job_id))
    
    # elif "submit-form" in request.form:
    #     # THIS DOES NOT WORK. Keeping the file content in the field seems to be disallowed as a security feature.
    #     form.file.data = request.files['file']

    return render_template("index.html", form=form, mergeform=mergeform)

def postprocess_grammar(result, grammar):

    extracted_data = []

    # Iterate over each report and its associated data
    for report, info in result.items():
        # Get the first key in the dictionary (here assumed to be the relevant field)
        first_key = next(iter(info))
        
        # Extract the content of the first field
        content = info.get(first_key, {}).get('content', '')
        
        # Parse the content string into a dictionary
        info_dict = ast.literal_eval(content)
        
        # Construct a dictionary containing the report and extracted information
        extracted_info = {'report': report}
        for key, value in info_dict.items():
            extracted_info[key] = value
        
        # Append the extracted information to the list
        extracted_data.append(extracted_info)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(extracted_data)

    # Reorder the columns to have 'report' as the first column
    columns = ['report'] + [col for col in df.columns if col != 'report']
    df = df[columns]

    print(df)
    return df


@app.route("/go", methods=["POST"])
def go():
    return 400
    # check if the post request has the file part
    if "file" not in request.files:
        flash("No part in the post request!", "danger")
        return redirect(url_for('index'))

    file = request.files["file"]

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == "":
        flash("No file selected!", "danger")
        # return "No selected file.", 400
        return redirect(url_for('index'))

    model_dir = Path(args.model_path)

    model_path = model_dir / request.form["model"]
    assert model_path.absolute().parent == model_dir

    # Check if model_path file exists
    if not model_path.is_file():
        flash(f"Model {request.form['model']} does not exist!", "danger")
        return redirect(url_for('index'))

    if file.filename.endswith('.csv'):
        try:
            print(file)
            df = pd.read_csv(file)
        except pd.errors.ParserError as e:
            # print the error message in console
            print(e)
            print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed. Usually, this is caused by a line break in a field. The file will be fixed and then read again.")
            # fix the file
            fixed_file = io.BytesIO()
            read_and_save_csv(file, fixed_file)
            fixed_file.seek(0)
            df = pd.read_csv(fixed_file)
    elif file.filename.endswith('.xlsx'):
        try:
            df = pd.read_excel(file)
            print(df.head())
            # ValueError: Excel file format cannot be determined, you must specify an engine manually.
        except ValueError as e:
            print(e)
            print("The error message indicates that the Excel file format cannot be determined. This means that the Excel file is not properly formatted and needs to be fixed. The file will be fixed and then read again.")
            # fix the file
            flash("Excel file is not properly formatted!", "danger")
            return redirect(url_for('index'))


    variables = [var.strip() for var in request.form["variables"].split(",")]
    job_id = secrets.token_urlsafe()
    global jobs
    jobs[job_id] = executor.submit(
        extract_from_report,
        df=df,
        model_name=request.form["model"],
        prompt=request.form["prompt"],
        symptoms=variables,
        temperature=float(request.form["temperature"]),
        pattern=request.form["pattern"],
        default=request.form["default_answer"],
        grammar=request.form["grammar"].replace("\r\n", "\n"),
    )

    return redirect(url_for('result', job=job_id))



@app.route("/result")
def result():
    jobid = request.args.get("job")
    job = jobs[jobid]

    if job.cancelled():
        return render_template('result.html', status="Job was cancelled")
    elif job.running():
        return render_template('result.html', status="Job is running, come back later (and refresh the page)")
    elif job.done():
        try:
            result_df = job.result()
        except Exception as e:
            flash("Postprocessing failed, probably the model did not generate anything.", "danger")
            return render_template('result.html', status="Job failed, error: " + str(e))

        result_io = io.BytesIO()
        result_df.to_csv(result_io, index=False)
        result_io.seek(0)
        return send_file(
            result_io,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{jobid}.csv",
        )
    else:
        return render_template('result.html', status="Job in queue, come back later (and refresh the page)")


def extract_from_report(
        df: pd.DataFrame,
        model_name: str,
        prompt: str,
        symptoms: Iterable[str],
        temperature: float,
        pattern: str,
        default: str,
        grammar: str,
        postprocess_type: PostProcessType
) -> dict[Any]:
    # Start server with correct model if not already running
    model_dir = Path(args.model_path)

    model_path = model_dir / model_name
    assert model_path.absolute().parent == model_dir

    global server_connection, current_model
    if current_model != model_name:
        server_connection and server_connection.kill()
        server_connection = subprocess.Popen(
            [
                args.server_path,
                "--model",
                str(model_path),
                "--ctx-size",
                str(args.ctx_size),
                "--n-gpu-layers",
                str(args.n_gpu_layers),
                # "--verbose",
            ],
        )
        current_model = model_name
        time.sleep(10)

    for _ in range(16):
        # wait until server is running
        try:
            requests.post(
                url="http://localhost:8080/completion",
                json={"prompt": "foo", "n_predict": 1}
            )
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)
    
    try:
        requests.post(
            url="http://localhost:8080/completion",
            json={"prompt": "foo", "n_predict": 1}
        )
    except requests.exceptions.ConnectionError:
        socketio.emit('load_failed')
        return
    
    socketio.emit('load_complete')

    results = {}

    socketio.emit('progress_update', {'progress': 0, 'total_steps': len(df.report)})

    for i, report in enumerate(df.report):
        print("parsing report: ", i)
        for symptom in symptoms:
            result = requests.post(
                url="http://localhost:8080/completion",
                json={
                    "prompt": prompt.format(
                        symptom=symptom, report="".join(report)
                    ),
                    "n_predict": args.n_predict,
                    "temperature": temperature,
                    "grammar": grammar,
                },
                timeout=60 * 20,
            )
            summary = result.json()
            if report not in results:
                results[report] = {}
            results[report][symptom] = summary
        #yield f"data: {i / total_reports * 100}\n\n"
        print(f"Report {i} completed.")
        socketio.emit('progress_update', {'progress': (i + 1) / len(df.report) * 100, 'total_steps': len(df.report)})

    socketio.emit('progress_complete', {'total_steps': len(df.report)})

    if postprocess_type.value == PostProcessType.DEFAULT.value:
        return postprocess(results, pattern, default)
    elif postprocess_type.value == PostProcessType.GRAMMAR.value:
        return postprocess_grammar(results, symptoms)
    else:
        socketio.emit('postprocess_failure', {'message': "No Postprocessing method found."})
        return False

@app.route("/merge", methods=["POST"])
def merge():
    files = request.files.getlist("files")
    merged_data = []

    for file in files:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            merged_data.append(df)
        elif file.filename.endswith('.pdf'):
            print(file.filename)
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
                print("pdf",page.extract_text())
            merged_data.append(pd.DataFrame({'report': [text]}))
        elif file.filename.endswith('.txt'):
            text = file.read().decode()
            print("text", text)
            merged_data.append(pd.DataFrame({'report': [text]}))

    merged_df = pd.concat(merged_data)
    merged_csv = merged_df.to_csv(index=False)
    return Response(
        merged_csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=merged.csv"},
    )

def postprocess(data, pattern: str, default: str) -> pd.DataFrame:
    symptoms = list(next(iter(data.values())).keys())

    result_dict = {
        "report": [],
    }
    for symptom in symptoms:
        result_dict[f"{symptom}_content"] = []
        result_dict[f"{symptom}_match"] = []

    for report, report_data in data.items():
        result_dict["report"].append(report)
        for symptom, llama_response in report_data.items():
            content = llama_response["content"]
            result_dict[f"{symptom}_content"].append(content)
            primary_matches = re.findall(pattern, content, re.IGNORECASE)
            result_dict[f"{symptom}_match"].append(
                primary_matches[0] if primary_matches else default
            )

    return pd.DataFrame.from_dict(result_dict)

import analysis


@app.route('/upload', methods=['POST'])
def upload_file():
    output_csv = request.files['output_csv']
    groundtruth_csv = request.files['groundtruth_csv']
    variables = [var.strip() for var in request.form["variables"].split(",")]
    plot = analysis.compare_and_plot(output_csv, groundtruth_csv, variables)
    return send_file(plot, mimetype='image/png')

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')


@app.route('/progress')
def progress():
    def generate():
        for i in range(101):
            yield f"data:{i}\n\n"
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')


@socketio.on('connect')
def handle_connect():
    print("Client Connected")

@socketio.on('disconnect')
def handle_connect():
    print("Client Disconnected")

if __name__ == "__main__":
    socketio.run(app, debug=True, port=args.port)