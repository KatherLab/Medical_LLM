import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO

def compare_and_plot(output_csv, groundtruth_csv, variables):
    output_df = pd.read_csv(output_csv)
    groundtruth_df = pd.read_csv(groundtruth_csv)

    for variable in variables:
        output_variable = output_df[variable + '_match']
        groundtruth_variable = groundtruth_df[variable]

        # Map German words to boolean values and 'fehlt' to None
        output_variable = output_variable.map({'Ja': True, 'nein': False, 'fehlt': None})
        # Handle 'Unknown' by deciding how to treat these cases. Here it's converted to a third category
        output_variable = output_variable.fillna('Unknown')

        # Ensure groundtruth_variable has the same categories
        # Since groundtruth does not have 'Unknown', this step might be unnecessary
        # But if you decide to handle 'Unknown' differently, adjust here
        # Convert all categories to strings to handle them consistently
        output_variable = output_variable.map({True: 'True', False: 'False', None: 'Unknown'}).fillna('Unknown')
        groundtruth_variable = groundtruth_variable.map({True: 'True', False: 'False'}).fillna('Unknown')

        # Print for debugging; can be removed in production code
        print("Output Variable Sample:")
        print(output_variable.head(50))
        print("Ground Truth Sample:")
        print(groundtruth_variable.head(50))

        # transfer output_variable type to boolean
        # Compute the confusion matrix with all possible labels
        labels = ['True', 'False', 'Unknown']
        cm = confusion_matrix(groundtruth_variable, output_variable, labels=labels)
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure()  # Create a new figure for each variable's plot to avoid overlap
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
        plt.title(f'Confusion Matrix for {variable}')
        plt.colorbar()
        tick_marks = range(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plot_file = BytesIO()
        plt.savefig(plot_file, format='png')
        plot_file.seek(0)

        # Here, you might want to adjust how you handle returning or saving the plot file,
        # especially if dealing with multiple variables in a loop
        return plot_file
