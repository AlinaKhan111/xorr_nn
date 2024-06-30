import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.models import load_model

# Load your trained model
model = load_model('path_to_your_model.h5')  # Adjust with your actual model loading code

# Load your test data
test_data = pd.DataFrame(data={"X1": [0, 0, 1, 1], "X2": [0, 1, 0, 1], "Y": [0, 1, 1, 0]})  # Example test data
X_test = test_data[['X1', 'X2']].values
Y_test = test_data['Y'].values

# Perform predictions
predictions = model.predict(X_test)

# Convert predictions to binary classes (if needed)
binary_predictions = np.round(predictions).astype(int)

# Calculate metrics
accuracy = accuracy_score(Y_test, binary_predictions)
precision = precision_score(Y_test, binary_predictions)
recall = recall_score(Y_test, binary_predictions)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Optionally, print predictions for each input
for i in range(len(X_test)):
    input_data = X_test[i]
    predicted_output = predictions[i][0]  # Adjust based on your model output shape
    print(f"Input: {input_data}, Predicted Output: {predicted_output}")
