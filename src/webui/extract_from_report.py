# %%
import json

import pandas as pd
import requests
from tqdm import tqdm

####################################################
# BEFORE RUNNING, DEFINE output_json, prompt
####################################################
# %%
output_json = "results-llama_german_70b_NRAD_blutung.jsonl"

grammar = r"""
root ::= allrecords

allrecords ::= (
  "{" ws "\"Datum\":" ws ("null" | date) ws "}"
  ws
)

date ::= (\d{2}\.\d{2}\.\d{4})

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

value  ::= object | array | string | number | ("true" | "false" | "null") ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n])?
"""

prompt = """[INST] <<SYS>>
Sie sind ein medizinischer Assistent von OpenAI ChatGPT. Im Folgenden finden Sie einen Bericht einer neuroradiologischen Intervention. Manchmal sind nicht alle Informationen angegeben. Wenn Informationen fehlen, geben Sie "null" an. Bitte extrahieren Sie die gesuchten Informationen aus dem Interventionsbericht. 
<</SYS>>
[/INST]
Beispiel: 
Untersuchung vom 22.12.2022.

Indikation: Notfallintervention ohne vorherige schriftliche Einverständniserklärung bei rechtshemisphärieller Schlaganfallsymptomatik mit CT-angiographisch nachgewiesenem prox. M2-Verschluss bei wohl thrombosiertem Aneurysma, ASPECT = 10. Thrombektomie frustran. IV Lyse erfolgt. NihSS: 5. 

Kam es zu einer Blutung? Ja, wenn eine Blutung oder SAB beschrieben wird. Nein, wenn im Befund steht, dass keine Blutung vorliegt. 
ASSISTANT: nein

[INST]
Das ist der Interventionsbericht:
{}
Kam es zu einer Blutung? Ja, wenn eine Blutung oder SAB beschrieben wird. Nein, wenn im Befund steht, dass keine Blutung vorliegt. 
[/INST]"""
'''
# %%
df = pd.read_excel("/mnt/bulk/isabella/llamaproj/NRAD/TextFilesContentOnly.xlsx")
# df = df[:10]
#print(df.head(10))

#%%
try:
    with open(output_json, "r") as outjson:
        lines_so_far = sum(bool(line.strip()) for line in outjson)
except FileNotFoundError:
    lines_so_far = 0

with open(output_json, "a") as outjson:
    for report in tqdm(df.content.iloc[lines_so_far:]):
        while True:
            try:
                result = requests.post(
                    url="http://localhost:8080/completion",
                    json={
                        "prompt": prompt.format("".join(report)),
                        "n_predict": 2048,
                        #"grammar": grammar,
                        "temperature": 0.1,
                    },
                )
                summary = result.json()
                break
            except json.decoder.JSONDecodeError:
                pass
        summary["report"] = report
        outjson.write(f"{json.dumps(summary, ensure_ascii=False)}\n")
        outjson.flush()
# %%
'''


def extract_from_report(output_json, prompt, symptoms, df_path):
    df = pd.read_excel(df_path)

    try:
        with open(output_json, "r") as outjson:
            lines_so_far = sum(bool(line.strip()) for line in outjson)
    except FileNotFoundError:
        lines_so_far = 0

    with open(output_json, "a") as outjson:
        for report in tqdm(df.content.iloc[lines_so_far:]):
            for symptom in symptoms:
                while True:
                    try:
                        result = requests.post(
                            url="http://localhost:8080/completion",
                            json={
                                "prompt": prompt.format(symptom, "".join(report)),
                                "n_predict": 2048,
                                "temperature": 0.1,
                            },
                        )
                        summary = result.json()
                        break
                    except json.decoder.JSONDecodeError:
                        pass
                summary["report"] = report
                outjson.write(f"{json.dumps(summary, ensure_ascii=False)}\n")
                outjson.flush()
