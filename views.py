import os
import numpy as np
import cv2
import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
import base64

from tensorflow.keras.models import load_model
from Project import settings


# Load the trained model once on import
model_path = os.path.join(settings.BASE_DIR, 'aiModel', 'model', 'breast_cancer_model.h5')
model = load_model(model_path)

class_labels = ["Benign", "Malignant", "Normal"]


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


import tensorflow as tf

def get_gradcam_heatmap(model, image_np, class_index):
    """
    model: your full Keras Sequential model
    image_np: numpy array with shape (1, 128, 128, 3) normalized float32 input
    class_index: int, predicted class index to explain
    """

    image = tf.convert_to_tensor(image_np, dtype=tf.float32)

    # Extract DenseNet base model
    densenet_model = model.layers[0]
    last_conv_layer_name = 'conv5_block16_concat'

    # Call once to make sure all internal outputs are built
    _ = densenet_model(image)

    # Get output of last conv layer
    conv_output_layer = densenet_model.get_layer(last_conv_layer_name).output

    # Build a model to get both the conv layer output and the model output
    grad_model = tf.keras.models.Model(
        inputs=densenet_model.input,
        outputs=conv_output_layer
    )

    with tf.GradientTape() as tape:
        # Get the conv outputs
        conv_outputs = grad_model(image)

        # Manually run rest of model (your classifier block)
        x = model.layers[1](conv_outputs)  # Flatten
        for layer in model.layers[2:]:
            x = layer(x)
        predictions = x

        # Compute loss for Grad-CAM
        loss = predictions[:, class_index]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight and sum channels
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original_image, 1 - alpha, 0)
    return superimposed_img




def predict_image(image):
    # Preprocess input image (BGR to RGB, resize, normalize)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (128, 128))
    norm_image = resized_image.astype("float32") / 255.0
    input_image = np.expand_dims(norm_image, axis=0)

    # Predict
    prediction = model.predict(input_image)
    class_index = np.argmax(prediction, axis=1)[0]
    result = class_labels[class_index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    # Advice dictionary
    advice = {
        "Malignant": "<b>A cancerous tumor</b> <span style='font-weight: lighter;'>that can grow and spread to other parts of the body if not treated promptly. <br>  <b style='color:#FF0059;'> Advice: </b> Please consult a doctor immediately.</span>",
        "Benign": "<b>A non-cancerous tumor</b> <span style='font-weight: lighter;'>that does not spread to other parts of the body. <br>  <b style='color:#FF0059;'> Advice:</b> Regular check-ups with a doctor are recommended for you.</span>",
        "Normal": "<b>Healthy breast tissue with no signs of cancer</b> <span style='font-weight: lighter;'> <br> <b style='color:#FF0059;'>Advice:</b> Keep up with self-checks and screenings.</span>"
    }

    # Generate Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, input_image, class_index)

    # Overlay heatmap on the original (full-size) image for visualization
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay_img = overlay_heatmap_on_image(original_rgb, heatmap)

    # Convert overlay to base64 string for JSON response
    overlay_base64 = encode_image_to_base64(cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    return result, advice[result], confidence, overlay_base64


@csrf_exempt
def predict_breast_cancer(request):
    doctors = []
    try:
        excel_path = os.path.join(settings.BASE_DIR, 'media', 'doctors-list.xlsx')
        df = pd.read_excel(excel_path, dtype=str)
        doctors = df.to_dict(orient='records')
    except Exception as e:
        # Log error - consider using logging instead of print
        print(f"Error reading Excel file: {e}")

    if request.method == "POST" and "image" in request.FILES:
        try:
            file = request.FILES["image"]
            image_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return JsonResponse({"error": "Invalid image uploaded."}, status=400)

            result, advice, confidence, heatmap_img_base64 = predict_image(image)

            return JsonResponse({
                "prediction": result,
                "advice": advice,
                "confidence": confidence,
                "doctors": doctors,
                "heatmap_image": heatmap_img_base64
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return render(request, "pages/prediction.html")
