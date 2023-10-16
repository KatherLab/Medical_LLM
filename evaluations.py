#%%
# This notebook compiles and tests the following evaluation metrics from llamaindex: 
# 1. Faithfulness Evaluator: To evaluate whether the reponse matches up with the source node.
# 2. Relevancy Evalautor:  To evaluate whether the respnse and the source node matches with the query.
# 3. Correctness Evaluator: To evaluate how much the response matches with your ideal answer (ground truth).
# 4. Embedding Similarity Evaluator: To evaluate the similarity score between embeddings of response and your ideal answer.
# 5. Guidedline Evaluator: to evaluate whether the generated reponse stands up to your guidelines.

#Imports

import nest_asyncio
import logging
import sys
import pandas as pd
import openai
from dotenv import load_dotenv
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
nest_asyncio.apply()
from IPython.display import display
from llama_index import (TreeIndex, 
                         VectorStoreIndex, 
                         SimpleDirectoryReader, 
                         ServiceContext, 
                         LLMPredictor,
                         Response)
from llama_index.llms import OpenAI
from llama_index.evaluation import FaithfulnessEvaluator
from llama_index.evaluation import RelevancyEvaluator
from llama_index.evaluation import CorrectnessEvaluator
from llama_index.evaluation import SemanticSimilarityEvaluator
from llama_index.evaluation import GuidelineEvaluator

pd.set_option("display.max_colwidth", 0)

#%%
# load key 
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

#%%
#create model instances
gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
# %%
#load documents
documents = SimpleDirectoryReader("./Data/").load_data()
#%%
# create vector index
service_context = ServiceContext.from_defaults(chunk_size=100)
vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

#%%
#user inputs
query_str = "what is the ASCO guideline?"
reference = "ASCO's clinical practice guidelines provide evidence-based recommendations and outline appropriate methods of treatment and care for clinicians. Our guidelines address specific clinical situations (disease-oriented) or the use of approved medical products, procedures, or tests (modality-oriented)."
guideline1 = "The response should fully answer the query."
guideline2 = "The response should avoid being vague or ambiguous."
guideline3 = "The response should be specific and use statistics or numbers when possible."
#%%
#Get the different evaluators
evaluator1 = FaithfulnessEvaluator(service_context=service_context_gpt4)
evaluator2 = RelevancyEvaluator(service_context= service_context_gpt4)
evaluator3 = CorrectnessEvaluator(service_context=service_context_gpt4)
evaluator4 = SemanticSimilarityEvaluator()
evaluator5 = GuidelineEvaluator(service_context=service_context_gpt4, guidelines=guideline1)
evaluator6 = GuidelineEvaluator(service_context=service_context_gpt4, guidelines=guideline2)
evaluator7 = GuidelineEvaluator(service_context=service_context_gpt4, guidelines=guideline3)

#%%
# Create result table 

nest_asyncio.apply()

def display_eval_df(response: Response, result1: str, result2: str, result3:str, result4:str, result5:str, result6:str, result7:str) -> None:
    if response.source_nodes == []:
        print("no response!")
        return None
    
    eval_df = pd.DataFrame(
        {
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Faithfulness Evaluator": "Pass" if result1.passing else "Fail",
            "Relevancy Evaluator": "Pass" if result2.passing else "Fail",
            "Correctness Evaluator": f"{result3.score}<br>{result3.feedback}",
            "Embedding Similarity Evaluator": f"{result4.score}<br>{result4.passing}",
            "Completeness Guideline Evaluator": f"{result5.passing}<br>{result5.feedback}",
            "Ambiguity Guideline Evaluator": f"{result6.passing}<br>{result6.feedback}",
            "Specificity Guideline Evaluator": f"{result7.passing}<br>{result7.feedback}",
        }, 
        index=[0],
    )

    # Apply global styles
    global_styles = {
        "inline-size": "400px",
        "overflow-wrap": "break-word",
        "text-align": "left",  # Align column content to the left
    }

    # Define CSS styles for column headers (column names)
    header_styles = {
        'selector': 'th',  # Apply styles to table headers (column names)
        'props': [
            ('text-align', 'left'),  # Align column headers to the left
        ]
    }
    styled_df = eval_df.style.set_properties(**global_styles).set_table_styles([header_styles])
#   styled_df = eval_df.style.set_properties(
#        **{ "text-align": 'left',
#           
#            "inline-size": "400px",
#            "overflow-wrap": "break-word",
#        },
#        subset=["Response", "Source", "Faithfulness Evaluator", "Relevancy Evaluator", "Correctness Evaluator", "Embedding Similarity Evaluator", "Completeness Guideline Evaluator", "Ambiguity Guideline Evaluator", "Specificity Guideline Evaluator"]
#    )

    display(styled_df)
    #return eval_df

#%%
# create responses for query and evaluate each of them 

query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)

result1 = evaluator1.evaluate_response(response=response_vector)
result2 = evaluator2.evaluate_response(query = query_str, response=response_vector)
result3 = evaluator3.evaluate(
    query=str(query_str),
    response=str(response_vector),
    reference=str(reference),
)
result4 = evaluator4.evaluate(
    response=str(response_vector),
    reference=str(reference),
)
result5 = evaluator5.evaluate(
    query = str(query_str),
    response = str(response_vector), 
    reference = str(reference), 
    )
result6 = evaluator6.evaluate(
    query = str(query_str),
    response = str(response_vector), 
    reference = str(reference), 
    )
result7 = evaluator7.evaluate(
    query = str(query_str),
    response = str(response_vector), 
    reference = str(reference), 
    ) 


eval_df = display_eval_df(response_vector, result1, result2, result3, result4, result5, result6, result7)

#%%
# save table to csv or json or text 

if eval_df is not None:
    # Save to CSV
    eval_df.to_csv('eval.csv', index=False, mode='a', header=not os.path.exists('eval.csv'))

    # Alternatively, save to JSON (you can use either 'records' or 'split' orient based on your preference)
    with open('eval.json', 'w') as json_file:
        json_file.write(eval_df.to_json(orient='records', lines=True))
        json_file.write('\n')  # Add newline between JSON records

    # Save to text
    with open('eval.txt', 'w') as file:
        file.write(str(response_vector))
        file.write('\n')
# %%
