import io
import re
import secrets
import subprocess
import time
from collections.abc import Iterable
from concurrent import futures
from pathlib import Path
from typing import Any, Optional
import pdfplumber

import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, send_file, Response, url_for
import PyPDF2

app = Flask(__name__)

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None
progress_updates: dict[JobID, list[str]] = {}

JobID = str
jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

app.config["DOWNLOAD_FOLDER"] = "/home/jeff/PycharmProjects/Medical_LLM/temp_output"
app.config["DB_PATH"] = "/home/jeff/LLM_database/mimicIV.db"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/go", methods=["POST"])
def go():
    # check if the post request has the file part
    if "file" not in request.files:
        return "No file part in the request.", 400

    file = request.files["file"]

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == "":
        return "No selected file.", 400

    model_dir = Path("/mnt/bulk/isabella/llamaproj")

    model_path = model_dir / request.form["model"]
    assert model_path.absolute().parent == model_dir

    df = pd.read_csv(file)
    variables = [var.strip() for var in request.form["variables"].split(",")]
    job_id = secrets.token_urlsafe()
    progress_updates[job_id] = []
    jobs[job_id] = executor.submit(
        extract_from_report,
        df=df,
        model_name=request.form["model"],
        prompt=request.form["prompt"],
        symptoms=variables,
        temperature=float(request.form["temperature"]),
        pattern=request.form["pattern"],
        default=request.form["default_answer"],
        jobid=job_id,
    )

    return redirect(url_for('result', job=job_id))


    return redirect(url_for('result', job=job_id))

@app.route("/result")
def result():
    jobid = request.args.get("job")
    print(jobid)
    print(jobs)
    job = jobs[jobid]

    if job.cancelled():
        return render_template('result.html', status="Job was cancelled")
    elif job.running():
        return Response(job.result()[1], mimetype='text/event-stream')
    elif job.done():
        result_df, _ = job.result()
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
        jobid: JobID,
) -> dict[Any]:
    # Start server with correct model if not already running
    model_dir = Path("/mnt/bulk/isabella/llamaproj")

    model_path = model_dir / model_name
    assert model_path.absolute().parent == model_dir

    global server_connection, current_model
    if current_model != model_name:
        server_connection and server_connection.kill()
        server_connection = subprocess.Popen(
            [
                "/mnt/bulk/isabella/llamaproj/llama.cpp/server",
                "--model",
                str(model_path),
                "--ctx-size",
                "2048",
                "--n-gpu-layers",
                "100",
                "--verbose",
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

    results = {}
    total_reports = len(df.report)
    progress = []

    for i, report in enumerate(df.report):
        print("parsing report: ", i)
        for symptom in symptoms:
            result = requests.post(
                url="http://localhost:8080/completion",
                json={
                    "prompt": prompt.format(
                        symptom=symptom, report="".join(report)
                    ),
                    "n_predict": 2048,
                    "temperature": temperature,
                },
                timeout=60 * 5,
            )
            summary = result.json()
            if report not in results:
                results[report] = {}
            results[report][symptom] = summary
        progress_updates[jobid].append(f"data: {i / total_reports * 100}\n\n")

    return postprocess(results, pattern, default)
@app.route("/merge", methods=["POST"])
def merge():
    files = request.files.getlist("files")
    merged_data = []

    for file in files:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            merged_data.append(df)
        elif file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfFileReader(file)
            print("pdf_reader",pdf_reader.numPages)
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



@app.route("/progress/<jobid>")
def progress(jobid: JobID):
    progress = progress_updates.get(jobid, [])
    if progress:
        return progress[-1]
    else:
        return "data: 0\n\n"


if __name__ == "__main__":
    app.run(debug=True, port=5002)
