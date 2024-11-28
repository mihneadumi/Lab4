import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from NCF import NCF

# Load the training dataset
data = pd.read_csv("train_data.csv")
data = data[['userId', 'movieId', 'rating']]

data['userId'] = data['userId'].astype('category').cat.codes
data['movieId'] = data['movieId'].astype('category').cat.codes

# Split data into training and validation sets
train, validation = train_test_split(data, test_size=0.2, random_state=42)

# Hyperparameters
num_users = data['userId'].nunique()
num_items = data['movieId'].nunique()

# Instantiate and compile the model
model = NCF(num_users, num_items)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Prepare training data
train_x = train[['userId', 'movieId']].values
train_y = (train['rating'] >= 3).astype(int).values

# Prepare validation data
validation_x = validation[['userId', 'movieId']].values
validation_y = (validation['rating'] >= 3).astype(int).values

history = model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=10,  # Increase epochs to let the model train longer
    validation_data=(validation_x, validation_y),
)

# Plot the training and validation loss with better visuals
plt.figure(figsize=(10, 6))  # Larger plot
plt.plot(history.history['loss'], label='Train Loss', color='blue', linestyle='-', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--', marker='x')

# Setting y-axis to log scale
plt.yscale('log')

# Adding grid and labels
plt.title('Training vs. Validation Loss (Log Scale)')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.grid(True)
plt.legend(loc='upper right')

# Save the plot first
plt.savefig('results/plot.png')

# Show the plot after saving
plt.show()

# Prepare test data (replace with your test dataset)
test = pd.read_csv("test_data.csv")

# Map userId and movieId in the test data to the encoded values
user_mapping = {k: v for k, v in enumerate(data['userId'].astype('category').cat.categories)}
item_mapping = {k: v for k, v in enumerate(data['movieId'].astype('category').cat.categories)}

test['userId'] = test['userId'].map(user_mapping)
test['movieId'] = test['movieId'].map(item_mapping)

# Drop rows with unmapped userId or movieId
test = test.dropna(subset=['userId', 'movieId'])

# Prepare the test data
test_x = test[['userId', 'movieId']].values.astype(int)
test_y = (test['rating'] >= 3).astype(int).values

# Predict on the test data
predictions = model.predict(test_x)


predicted_classes = (predictions >= 0.5).astype(int)

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_y, predicted_classes)

precision = precision_score(test_y, predicted_classes)
recall = recall_score(test_y, predicted_classes)
f1 = f1_score(test_y, predicted_classes)

# Save these metrics to a text file
with open('results/classification_report.txt', 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write("Precision (also known as Positive Predictive Value) measures the ability of the model to not label a negative sample as positive")
    f.write(f"Recall: {recall}\n")
    f.write("Recall (also known as Sensitivity or True Positive Rate) measures the ability of the model to find all the positive instances in the dataset")
    f.write(f"F1 Score: {f1}\n")
    f.write("F1 Score is the harmonic mean of Precision and Recall. It is a single metric that combines both Precision and Recall")

# Save the confusion matrix to a text file
with open('results/confusion_matrix.txt', 'w') as f:
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# Compute RMSE
rmse = mean_squared_error(test_y, predictions, squared=False)
print("Calculating RMSE (a measure of the average error between predicted and actual values). 0 is perfect prediction, while 1 is as good as a random guess.")
print(f"Test RMSE: {rmse}")

# save test results to results/
with open('results/test_results.txt', 'w') as f:
    f.write(f"Test RMSE: {rmse}")

    # Save accuracy metrics to the results
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    with open('results/training_accuracy.txt', 'w') as f:
        f.write("Epoch\tTraining Accuracy\n")
        for epoch, acc in enumerate(train_accuracy):
            f.write(f"{epoch + 1}\t{acc}\n")

    # Plot accuracy over epochs
    plt.figure(figsize=(10, 6))  # Larger plot
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', marker='x')

    # Adding grid and labels
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')

    # Save the plot first
    plt.savefig('results/accuracy_plot.png')

    # Show the plot after saving
    plt.show()

    with open('results/validation_accuracy.txt', 'w') as f:
        f.write("Epoch\tValidation Accuracy\n")
        for epoch, acc in enumerate(val_accuracy):
            f.write(f"{epoch + 1}\t{acc}\n")