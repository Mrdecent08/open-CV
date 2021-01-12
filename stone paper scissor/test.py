from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('model.h5')
print("Model Loaded Successfully")
print("Loaded model from disk")
def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (256, 256),grayscale=True)

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    max = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["0", "1", "2"]
    result = classes[max_prob-1]
    print(img_name,result)



import os
path = 'D:/Artificial Intelligence/stonepaperscissor/datasets/test'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpg' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')