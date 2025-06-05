# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Essential imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
# Plot style
sns.set_style('darkgrid')
plt.style.use('default')
# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 30
TRAIN_DIR = '/content/drive/MyDrive/dataset1/Data/train'
# Use ImageDataGenerator with data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,
 rotation_range=25,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
# Learning rate schedule
def exponential_decay_fn(epoch):
    return 0.001 * np.exp(-0.1 * epoch)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# Build VGG19 model
base_model = tf.keras.applications.VGG19(
include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)
for layer in base_model.layers:
    layer.trainable = False
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
#  Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler],
    verbose=1
)
# Save the model
model.save('/content/drive/MyDrive/VGG19_model.h5')
# Plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train')
plt.plot(epochs_range, val_acc, label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train')
plt.plot(epochs_range, val_loss, label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
# Evaluate the model
val_generator.reset()
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))
#  ROC AUC Score for one-vs-rest
fpr = {}
tpr = {}
roc_auc = {}
for i in range(train_generator.num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc[i] = roc_auc_score(y_true == i, y_pred_probs[:, i])
# Plot ROC curve
plt.figure(figsize=(8, 6))
for i, label in enumerate(val_generator.class_indices.keys()):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Multi-class ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
!pip install flask flask-ngrok pyngrok
import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pyngrok import ngrok, conf
from google.colab import drive
drive.mount('/content/drive')
# Set ngrok auth token
conf.get_default().auth_token = "2wIyE0FFv7y48hxM0NncrgJ14hx_6WgHvEsQKh7GaJqg7TFXq"
# Load model
model_path = "/content/drive/MyDrive/VGG19_model.h5"
model = load_model(model_path)
IMAGE_SIZE = (256, 256)
class_indices = {0: "Mild Impairment", 1: "Moderate Impairment", 2: "No Impairment", 3: "Very Mild Impairment"}
app = Flask(__name__)
# HTML template with uploaded image display
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Alzheimer MRI Stage Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f1f1f1;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 50px;
        }
        .box {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
            text-align: center;
        }
        input[type=file], input[type=submit] {
            margin: 15px 0;
            padding: 10px;
            width: 80%;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        input[type=submit] {
            background: #4CAF50;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
 input[type=submit]:hover {
            background: #45a049;
        }
        .note {
            font-size: 0.95em;
            color: #ff5733;
            margin-top: 10px;
        }
        .preview {
            margin-top: 20px;
        }
        .preview img {
            max-width: 256px;
            border-radius: 10px;
            border: 1px solid #999;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>Alzheimer MRI Stage Predictor</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <div class="note">Please ensure the MRI scan is captured under white light for best prediction results.</div>
 <br>
            <input type="submit" value="Predict Stage">
        </form>
        {% if prediction %}
            <h2>Predicted Stage: <span style="color: green;">{{ prediction }}</span></h2>
        {% endif %}
        {% if image_url %}
            <div class="preview">
                <h3>Uploaded Image:</h3>
                <img src="{{ image_url }}" alt="Uploaded MRI">
            </div>
        {% endif %}
    </div>
</body>
</html>
'''
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    image_url = None
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            os.makedirs('static', exist_ok=True)
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred)
            prediction = class_indices[predicted_class]
            image_url = url_for('static', filename=file.filename)

    return render_template_string(html_template, prediction=prediction, image_url=image_url)
# Start ngrok tunnel and Flask app
public_url = ngrok.connect(5000)
print(f"🔗 Public URL: {public_url}")
app.run(port=5000)