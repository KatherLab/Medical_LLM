from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/update-grammar', methods=['POST'])
def update_grammar():
    content = request.json

    features = content.get('features', [])
    additional_records = content.get('additional_records', {})

    # Compose grammar rules based on user input
    records_grammar = ",\n  ".join([f'ws "\"{feature}\":" ws record' for feature in features])
    additional_records_grammar = "\n".join(
        f'record_{key} ::= (\n'
        f'    "{{"\n'
        f'    ws "\"excerpt\":" ws ( string | "null" ) ","\n'
        f'    ws "\"present\":" ws ("true" | "false") ws\n'
        f'    ws "}}"\n'
        f'    ws\n'
        f')' for key in additional_records
    )

    # Complete grammar including user-defined features and records
    grammar = f"""
root ::= allrecords
value ::= object | array | string | number | ("true" | "false" | "null") ws

allrecords ::= (
  {{
  {records_grammar}
  ws "}}"
  ws
)

{additional_records_grammar}

object ::=
  "{{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}}" ws

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

ws ::= ([ \t\n])?
"""

    return jsonify({'updated_grammar': grammar})


if __name__ == '__main__':
    app.run(debug=True)
