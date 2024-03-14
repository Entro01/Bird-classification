from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
#from torchvision import transforms
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os

model = load_model('densenet_model.h5')
for layer in model.layers:
    layer._name = layer.name.replace('/', '_') # Replace '/' with '_'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return predict(filename)
    return render_template('upload.html')

def predict(image_path):
    img = Image.open(image_path)
    img_t = transform(img)
    batch = torch.unsqueeze(img_t, 0)
    output = model(batch)
    _, predicted = torch.max(output, 1)
    return render_template('result.html', prediction=predicted.item())

if __name__ == '__main__':
    app.run(debug=True)
