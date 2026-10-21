from pathlib import Path
import tensorflow as tf

# Run this file from the project root:
# python train.py

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pothole_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 123
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 5

CLASS_NAMES_EXPECTED = ["no_pothole", "pothole"]


def count_images(folder: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


if not DATA_DIR.exists():
    raise SystemExit(
        "Dataset folder not found. Create dataset/no_pothole and "
        "dataset/pothole, add labeled images, then run python train.py."
    )

counts = {
    name: count_images(DATA_DIR / name)
    for name in CLASS_NAMES_EXPECTED
}

print("Dataset image counts:")
for name, count in counts.items():
    print(f"  {name}: {count}")

missing = [name for name in CLASS_NAMES_EXPECTED if not (DATA_DIR / name).is_dir()]
if missing:
    raise SystemExit(
        f"Missing class folder(s): {missing}. Expected folders: "
        "dataset/no_pothole/ and dataset/pothole/."
    )

if any(counts[name] < 2 for name in CLASS_NAMES_EXPECTED):
    raise SystemExit(
        "\nTraining stopped: both classes need at least 2 valid images, "
        "and preferably hundreds of varied images per class.\n"
        "Put normal road images in dataset/no_pothole/ and images with "
        "visible potholes in dataset/pothole/. Do not put the same image "
        "in both folders."
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=CLASS_NAMES_EXPECTED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=CLASS_NAMES_EXPECTED,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(500, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# IMPORTANT:
# Raw image pixels (0..255) enter the model. Preprocessing is INSIDE
# the model, so app.py must NOT preprocess the image a second time.
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="road_image")
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pothole_probability")(x)
model = tf.keras.Model(inputs, outputs)

metrics = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=metrics,
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]

print("\nTraining classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
)

print("\nFine-tuning final MobileNetV2 layers...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
# Keep BatchNormalization layers frozen for stable small-dataset fine-tuning.
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=metrics,
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)

# Load the best checkpoint selected on validation loss and evaluate it.
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)

print("\nFinal validation evaluation:")
results = model.evaluate(val_ds, return_dict=True, verbose=1)
for name, value in results.items():
    print(f"{name}: {value:.4f}")

print(f"\nSaved model: {MODEL_PATH}")
print(
    "Reminder: validation results are estimates. Test on a separate set "
    "of road images that was not used for training or validation."
)
