
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


IMAGE_SIZE = (150, 150)
IMAGE_SHAPE = (150, 150, 3)
BATCH_SIZE = 16
NUM_EPOCHS = 200
NUM_FIREFLIES = 10
MAX_ITERATIONS = 5
RANDOM_SEED = 42

TRAIN_DIR = "/kaggle/input/brain-tumour-classification/BrainTumor_1/Train"
TEST_DIR = "/kaggle/input/brain-tumour-classification/BrainTumor_1/Test"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_image_dataframe(directory):

    filepaths = []
    labels = []

    for folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)

        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, filename)

            if os.path.isfile(image_path):
                filepaths.append(image_path)
                labels.append(folder)

    return pd.DataFrame({
        "image_data": filepaths,
        "label": labels
    })


# Load training and test datasets
train_df = load_image_dataframe(TRAIN_DIR)
test_df = load_image_dataframe(TEST_DIR)

# Split training data to create a separate validation set
train_df, valid_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=train_df["label"]
)

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(valid_df)}")
print(f"Testing images:    {len(test_df)}")

print("\nClass distribution:")
print(train_df["label"].value_counts())


def create_data_generators(train_df, valid_df, test_df):

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_data",
        y_col="label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=True,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED
    )

    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col="image_data",
        y_col="label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_data",
        y_col="label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=False,
        batch_size=BATCH_SIZE,
        classes=list(train_generator.class_indices.keys())
    )

    return train_generator, valid_generator, test_generator


train_generator, valid_generator, test_generator = (
    create_data_generators(train_df, valid_df, test_df)
)

NUM_CLASSES = train_generator.num_classes

print(f"\nNumber of classes: {NUM_CLASSES}")
print("Class indices:", train_generator.class_indices)


def create_model(num_classes=NUM_CLASSES):

    model = models.Sequential([
        layers.Input(shape=IMAGE_SHAPE),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def compile_model(model, learning_rate):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def firefly_algorithm(
    train_generator,
    valid_generator,
    num_fireflies=NUM_FIREFLIES,
    max_iter=MAX_ITERATIONS,
    search_epochs=10
):
    

    fireflies = [
        {
            "lr": np.random.uniform(1e-5, 1e-1),
            "batch_size": int(np.random.randint(16, 128))
        }
        for _ in range(num_fireflies)
    ]

    best_params = None
    best_accuracy = -np.inf

    for iteration in range(max_iter):
        print(f"\nFirefly iteration {iteration + 1}/{max_iter}")

        for index, firefly in enumerate(fireflies):
            print(
                f"  Firefly {index + 1}/{num_fireflies}: "
                f"lr={firefly['lr']:.6f}, "
                f"batch_size={firefly['batch_size']}"
            )

            # Use the candidate batch size for training.
            train_generator.batch_size = firefly["batch_size"]
            valid_generator.batch_size = firefly["batch_size"]

            model = compile_model(
                create_model(),
                firefly["lr"]
            )

            history = model.fit(
                train_generator,
                validation_data=valid_generator,
                epochs=search_epochs,
                verbose=0
            )

            accuracy = max(history.history["val_accuracy"])
            firefly["accuracy"] = accuracy

            print(f"    Validation accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "lr": firefly["lr"],
                    "batch_size": firefly["batch_size"],
                    "accuracy": accuracy
                }

            # Clear the model before evaluating the next candidate.
            del model
            tf.keras.backend.clear_session()

        # Rank candidates by validation accuracy.
        fireflies.sort(
            key=lambda firefly: firefly["accuracy"],
            reverse=True
        )

        print(
            f"Best accuracy so far: {best_accuracy:.4f}"
        )

    # Restore the original batch size after the search.
    train_generator.batch_size = BATCH_SIZE
    valid_generator.batch_size = BATCH_SIZE

    return best_params


best_params = firefly_algorithm(
    train_generator,
    valid_generator,
    num_fireflies=NUM_FIREFLIES,
    max_iter=MAX_ITERATIONS,
    search_epochs=10
)

print("\nBest hyperparameters:")
print(best_params)



# Apply the selected batch size.
train_generator.batch_size = best_params["batch_size"]
valid_generator.batch_size = best_params["batch_size"]

model = compile_model(
    create_model(),
    best_params["lr"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_brain_tumor_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1
)


test_generator.reset()

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")



def plot_training_history(history):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_training_history(history)


test_generator.reset()

predictions = model.predict(test_generator, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.classes

class_names = [
    name
    for name, index in sorted(
        test_generator.class_indices.items(),
        key=lambda item: item[1]
    )
]

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)
ax.set_title("Brain Tumor Classification - Confusion Matrix")
plt.tight_layout()
plt.show()