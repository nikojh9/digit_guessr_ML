from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load din trænet model
model = load_model("digit_guessr_v1.h5")

# forbered billede så den passer til modellens træning
def prepare_image(file_path):
    img = Image.open(file_path).convert("L")          # gråskala
    img = img.resize((28,28))                         # resize
    img = ImageOps.invert(img)                        # invert farver
    img = np.array(img)/255.0                          # normaliser
    img = img.reshape(1,28,28,1)                      # til model input
    return img

#test billede
img = prepare_image("assets//testImages/billede5.jpg")
prediction = model.predict(img)
digit = np.argmax(prediction)
print(f"Modellen gætter: {digit}")