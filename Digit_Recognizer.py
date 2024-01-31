import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")

# Assuming 'label' is the name of the target variable column in your dataset
X = data.drop(columns=['label'])
y = data['label']

# One-hot encode the target variable
y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40275478)

X_train, X_test = X_train / 255., X_test / 255.

tf.random.set_seed(40265478)

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])

    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    
    model.add(tf.keras.layers.Dense(10, activation="softmax"))  # Assuming 10 classes
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="digit_recognizer", project_name="my_reognizer", seed=40265478)

# Corrected validation_data parameter
random_search_tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

top3_parms = random_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_parms[0].values)

best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary())

test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test data
test = pd.read_csv("test.csv")
pred_probabilities = best_model.predict(test)

# Convert predicted probabilities to predicted labels
pred_labels = pred_probabilities.argmax(axis=1)

# Create a DataFrame for submission
submission = pd.DataFrame({"ImageID": test.index + 1, "Label": pred_labels})

# Save the submission DataFrame to a CSV file
submission.to_csv("submission.csv", index=False)
print("Done")