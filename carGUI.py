from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model

classifier = Sequential()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('car_data_bmw/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
model = load_model(
    "carModelBMW.h5",
    custom_objects=None,
    compile=True
)

while(True) :
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/Users/sohitgatiganti/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)
    test_image = image.load_img(root.filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict_proba(test_image)
    print(training_set.class_indices)
    maxIndex=0
    maxVal=result[0][0]
    prediction="Nothing"
    for i in range(0, len(result[0])): 
        if result[0][i] > maxVal:
            maxVal=result[0][i]
            maxIndex=i
    print(maxIndex)
    print(maxVal)
    for name, number in training_set.class_indices.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if number == maxIndex:
            prediction = name
    print(prediction)
    original = Image.open(root.filename)
    render = ImageTk.PhotoImage(original)
    img = Label(image = render)
    img.image = render
    img.place(x=0,y=0)
    text = Label(text=prediction, anchor=S)
    text.place(x=50,y=50)
    text.config(font=("Courier", 44))
    text.pack()
    root.title("Image Classifier")
    root.geometry(str((render.width()+10)) + "x" + str((render.height()+10)))
    root.mainloop()