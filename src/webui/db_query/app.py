from flask import Flask, jsonify, request, send_file, render_template
import sqlite3
from flask import current_app
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/jeff/PycharmProjects/Medical_LLM/temp_output'
app.config['DB_PATH'] = '/home/jeff/LLM_database/mimicIV.db'


@app.route('/')
def index():
    return render_template('db_query.html')
# Database connection
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
        temp_dir = current_app.config['UPLOAD_FOLDER']  # Define this in your app config
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

if __name__ == '__main__':
    app.run(debug=True)
