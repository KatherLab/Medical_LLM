import io
import re
import secrets
import subprocess
import time
from collections.abc import Iterable
from concurrent import futures
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, send_file

app = Flask(__name__)

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None

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
    )

    return redirect(f"/result?job={job_id}")


@app.route("/result")
def result():
    jobid = request.args.get("job")
    job = jobs[jobid]

    if job.cancelled():
        return "Job was cancelled", 200
    elif job.running():
        return "Job is running, come back later (and refresh the page)", 200
    elif job.done():
        result_df = job.result()
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
        return "Job in queue, come back later (and refresh the page)", 200


def extract_from_report(
        df: pd.DataFrame,
        model_name: str,
        prompt: str,
        symptoms: Iterable[str],
        temperature: float,
        pattern: str,
        default: str,
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
    for report in df.report:
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

    return postprocess(results, pattern, default)


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


if __name__ == "__main__":
    app.run(debug=True, port=5001)
