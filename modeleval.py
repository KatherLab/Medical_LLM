# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from tqdm import trange

# %%
df = pd.read_csv("MIMIC_groundtruth.csv")
df.head()

#%%
# print(df[df['liver cirrhosis'] == False])


# %%
# symptoms = [
#         "ascites",
#         "abdominal pain",
#         "shortness of breath",
#         "confusion",
#         "liver cirrhosis",
#     ]
# df[df[symptoms[1]].isna()]
# %%
# %%
def parse(x):
    if x == 0:
        return "False"
    elif x == 1:
        return "True"
    else:
        return "None"


def result_json_to_df(json_path):
    symptoms = [
        "ascites",
        "abdominal pain",
        "shortness of breath",
        "confusion",
        "liver cirrhosis",
    ]
    with open(json_path, "r") as json_file:
        records = []
        for line in json_file:
            try:
                llama_response = json.loads(line)
                extracted_info = json.loads(llama_response["content"])
                records.append(
                    (
                        llama_response["report"],
                        *[
                            extracted_info[label] and extracted_info[label]["present"]
                            for label in symptoms
                        ],
                    )
                )
            except json.JSONDecodeError:
                continue

    pred_df = pd.DataFrame(records, columns=["report", *symptoms])
    pred_df[symptoms] = pred_df[symptoms].applymap(parse)
    return pred_df



# %%
# set(gt_df.report) - set(pred_df.report)
# %%
#predgpt_df = pd.read_csv("out_mimic_gpt3.5.csv", error_bad_lines=False)
# predgpt_df = pd.read_csv("out_mimic_gpt3.5.csv", error_bad_lines=False)
#predgpt_df.head()

#%%
gt_df = pd.read_csv("MIMIC_groundtruth.csv")
symptoms = [
    "ascites",
    "abdominal pain",
    "shortness of breath",
    "confusion",
    "liver cirrhosis",
]
# gt_df[symptoms] = gt_df[symptoms].map(parse)
gt_df[symptoms] = gt_df[symptoms].applymap(parse)



#%%
for symptom in symptoms:  # ["ascites"]:#
    models = []
    precision_lowers, precision_medians, precision_uppers = [], [], []
    recall_lowers, recall_medians, recall_uppers = [], [], []
    accuracy_lowers, accuracy_medians, accuracy_uppers = [], [], []

    for model in ["70b"]: #"13b", "70b"]:
        # pred_df = result_json_to_df(f"results-{model}_exct_answ.jsonl")
        # pred_df = result_json_to_df(f"results-{model}_oneshot.jsonl")
        # pred_df = result_json_to_df(f"results-{model}_oneshot_def.jsonl")
        pred_df = result_json_to_df(f"results-{model}_p1110_4.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        precisions, recalls, accuracies = [], [], []
        for _ in trange(10):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]

            # class (True,False, None)
            category = "True"
            precisions.append(precision_score(y_true == category, y_pred == category))
            recalls.append(recall_score(y_true == category, y_pred == category))
            accuracies.append(accuracy_score(y_true == category, y_pred == category))

        models.append(model)

        precision_lowers.append(np.quantile(precisions, 0.025))
        precision_medians.append(np.quantile(precisions, 0.5))
        precision_uppers.append(np.quantile(precisions, 0.975))

        recall_lowers.append(np.quantile(recalls, 0.025))
        recall_medians.append(np.quantile(recalls, 0.5))
        recall_uppers.append(np.quantile(recalls, 0.975))

        accuracy_lowers.append(np.quantile(accuracies, 0.025))
        accuracy_medians.append(np.quantile(accuracies, 0.5))
        accuracy_uppers.append(np.quantile(accuracies, 0.975))

        # confusion matrices
        plt.figure()
        y_true = df[symptom]
        y_pred = df[f"{symptom} pred"]
        ConfusionMatrixDisplay.from_predictions(
            y_true.map(str),
            y_pred.map(str),
            labels=["True", "False", "None"],
            normalize="true",  # normalize nach predicted oder true label
            #normalize="pred", 
            #values_format="1.2f",
        )
        plt.title(f"{symptom} {model}")
        plt.show()
        plt.close()

    plt.figure()
    plt.errorbar(
        range(len(models)),
        precision_medians,
        yerr=np.abs(np.stack([precision_lowers, precision_uppers]) - precision_medians),
        capsize=3,
        label="precision",
    )
    plt.errorbar(
        range(len(models)),
        recall_medians,
        yerr=np.abs(np.stack([recall_lowers, recall_uppers]) - recall_medians),
        capsize=3,
        label="recall",
    )
    plt.errorbar(
        range(len(models)),
        accuracy_medians,
        yerr=np.abs(np.stack([accuracy_lowers, accuracy_uppers]) - accuracy_medians),
        capsize=3,
        label="accuracy",
    )
    plt.xticks(np.arange(len(models)), labels=models)
    plt.title(f"{symptom} = {category}")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


# %%
# # Calculate accuracy
# # Extracting ground truth and predicted values
# y_true = sample_df['symptom']
# y_pred = sample_df['symptom pred']

# # Calculate and print the accuracy
# accuracy = accuracy_score(y_true, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# for symptom in symptoms:#["ascites"]:#
#     models = []
#     precision_lowers, precision_medians, precision_uppers = [], [], []
#     recall_lowers, recall_medians, recall_uppers = (
#         [],
#         [],
#         [],
#     )

#     for model in ["7b", "13b", "70b"]:
#         pred_df = result_json_to_df(f"results-{model}_exc_answ.jsonl")
#         df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])
#         y_true = sample_df[symptom]
#         y_pred = sample_df[f"{symptom} pred"]
#     accuracy = accuracy_score(y_true, y_pred)
#     print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
