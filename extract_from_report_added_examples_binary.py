# %%
import json

import requests
from tqdm import tqdm
import pandas as pd

output_json = "results-13b_binary_p1.jsonl"

grammar = r"""
root   ::= allrecords   # This line defines the starting rule, 'root',
value  ::= object | array | string | number | ("true" | "false" | "null") ws # The 'value' rule can be an object, array, string, number, or one of the boolean literals "true" or "false", or the literal "null". Each of these is followed by whitespace (ws).

allrecords ::= (  # This complex rule defines a specific structure consisting of a series of key-value pairs, each corresponding to different medical conditions or symptoms. Each key is followed by a 'record'.
  "{"
  ws "\"ascites\":" ws record ","
  ws "\"abdominal pain\":" ws record ","
  ws "\"shortness of breath\":" ws record ","
  ws "\"confusion\":" ws record ","
  ws "\"liver cirrhosis\":" ws record
  ws "}"
  ws
)

record ::= ( # Defines the structure of a 'record', which includes an "excerpt" (that can be a string or null) and a "present" field (a boolean)
    "{"
    ws "\"excerpt\":" ws ( string | "null" ) ","
    ws "\"present\":" ws ("true" | "false") ws 
    ws "}"
    ws
)

object ::= # Specifies the structure of an object, which is a set of string-value pairs enclosed in curly braces ({}). This rule allows for optional and multiple pairs.
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::= # Defines the structure of an array, which is a sequence of values enclosed in square brackets ([]). Values can be separated by commas.
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::= #Describes how strings are structured, including how to handle escape sequences (like \n for a newline).
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws #Outlines the format of a number, including integers, decimals, and scientific notation.

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n])? #This is a rule for optional whitespace, which can be a space, a tab, or a newline character. It's used throughout the grammar to allow for flexible formatting
"""
# prompt snippets
# Strictly give the information from the report the user requests. Strictly keep with the content of the report, do not make anything up. You will be provided with definitions, that help answering the questions. Do not answer the questions with content of the definitions. 

prompt = """[INST] <<SYS>>
You are programmed as a cooperative medical assistant. A patient report will be available to you, and users will request specific information from this report. Your responses should adhere rigorously to the information contained within the provided report, ensuring no fabrication or assumption of details not explicitly stated.  

<</SYS>>
This is the report:
{}
Now answer following questions:
From the report, is ascites present at or before patient admission? Provide an excerpt from text.
From the report, is abdominal pain present at or before admission? Provide an excerpt from the text.
From the report, is shortness of breath present at or before admission? Provide an excerpt from the text. 
From the report, is confusion present at or before admission? Provide an excerpt from the text. 
From the report, is liver cirrhosis present or suspected at admission? Provide an excerpt from the text. 

These are the definitions:
Ascites refers to the accumulation of excess fluid in the peritoneal cavity, which is the space between the organs and the abdominal wall, often resulting from liver disease, heart failure, or cancer. 
Abdominal pain refers to any discomfort or pain that occurs in the abdominal area. It may sometimes be abbreviated as "abd pain" in medical contexts. The pain can also be specifically located and described by its region: Epigastric: Near the upper-middle region of the abdomen. RUQ: Right Upper Quadrant. RLQ: Right Lower Quadrant. LUQ: Left Upper Quadrant. LLQ: Left Lower Quadrant. If a 10-point review of systems (ROS) does not indicate any issues (is described as negative) and "abdominal pain" or its abbreviation are not explicitly mentioned in a medical report, it indicates that the patient does not have abdominal pain per the context provided in the definition. 
Shortness of breath: Shortness of breath (also known as SOB or dyspnea) refers to difficulty breathing. If it occurs during physical activity, it's referred to as dyspnea on exertion (DOE). If a 10 point review of systems (ROS) is negative (i.e., does not indicate any abnormality or issue) and the terms "dyspnea," "SOB," or "DOE" are not otherwise mentioned in a medical report, this is taken to mean the patient is not experiencing shortness of breath according to the context given. 
Confusion is a mental state characterized by disorientation and an inability to think clearly, often manifesting as difficulty remembering, making decisions, and maintaining awareness of critical aspects such as time, place, and personal identity. In medical contexts, the concept of orientation is pivotal. 'Oriented x4' indicates that an individual is lucid and aware of four key domains: person (awareness of oneself), place (recognition of physical location), time (understanding of the day, date, and/or time), and situation (comprehension of the ongoing events or circumstances). Consequently, being 'Oriented x4' signifies the absence of confusion. Conversely, if orientation is noted as less than 4, e.g., 'oriented x3', confusion is presumed present. Furthermore, impaired vigilance, exemplified when a patient is only intermittently responsive, is also indicative of confusion. Practical examples from medical reports might include phrases such as 'pt has brief period of confusion' or 'alert-oriented x3', suggesting episodes or states of confusion within the patient's condition. 
Liver cirrhosis: Is a late stage of scarring (fibrosis) of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism, leading to loss of liver function and potential complications like bleeding, jaundice, and hepatic encephalopathy. Examples: HCV cirrhosis, decompensated alcoholic and Hepatitis C cirrhosis, ETO cirrhosis. 
[/INST]"""
# %%
df = pd.read_csv("MIMIC_groundtruth_TF.csv")

try:
    with open(output_json, "r") as outjson:
        lines_so_far = sum(bool(line.strip()) for line in outjson)
except FileNotFoundError:
    lines_so_far = 0

with open(output_json, "a") as outjson:
    for report in tqdm(df.report.iloc[lines_so_far:]):
        while True:
            try:
                result = requests.post(
                    url="http://localhost:8080/completion",
                    json={
                        "prompt": prompt.format("".join(report)),
                        "n_predict": 2048,
                        "grammar": grammar,
                        "temperature": 0.500000011920929,
                    },
                )
                summary = result.json()
                break
            except json.decoder.JSONDecodeError:
                pass
        summary["report"] = report
        outjson.write(f"{json.dumps(summary)}\n")
        outjson.flush()
# %%