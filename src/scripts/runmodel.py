import ast
import os
os.environ['TRANSFORMERS_CACHE'] = 'path to your desired cache directory'
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf" #name of your desired model (provided on HF)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# with open('path to your text files.txt', 'r') as file:
#     pmhist = [line.strip() for line in file]
#     content = pmhist[:2]

# print(pmhist[:2])

def extract_elements_from_file(filename):
    with open(filename, 'r') as f:
        # Read the entire content of the file
        content = f.read()
        
        # Split the content by "\n\n\n" to get the elements
        elements = content.split("\n\n\n")
        
        # Return the list of elements
        return elements

# Example usage:
filename = 'path to your text files.txt'
elements = extract_elements_from_file(filename)
content = elements[:2]



prompt = """Please extract the information from the text. Provide the text passage containing the information first, then answer the question. 
Ascites, excerpt from text: ""
Is ascites present at admission? Give only yes, no or not available:""
Abdominal pain, excerpt from text: ""
Is abdominal pain present at admission? Give only yes, no or not available:""
Shortness of breath, excerpt from text: ""
Is shortness of breath present at admission? Give only yes, no or not available:"" 
Confusion, excerpt from text: ""
Is confusion present at admission? Give only yes, no or not available:""
Liver cirrhosis, excerpt from text:""
Is liver cirrhosis present at admission? Give only yes, no or not available:""

"""

for c in content:
    print("----")
    prompt_template=f'''You are a helpful medical assistant. You will be provided with a patient report.The report has been extracted from a pdf file, so there might be errors. Strictly give the information in the format the user requests. Strictly keep with the content of the report, do not make anything up. If information is missing, give "not available".
    User: {prompt} This is the report: {c}
    Assistant: 

    '''


    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.00000000000000000000000000001, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     repetition_penalty=1.1
# )

# print(pipe(prompt_template)[0]['generated_text'])

