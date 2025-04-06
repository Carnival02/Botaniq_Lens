import streamlit as st 
import tensorflow as tf
import numpy as np
#Tensorflow Model prediction

def model_prediction(test_image):
    model=tf.keras.models.load_model("C:\\Users\\harsh\\OneDrive\\Desktop\\WEbAPP\\trained_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)









#Sidebar
st.sidebar.title('Dashboard')
app_mode=st.sidebar.selectbox("Select page",["Home","About","Prediction"])
  
  #main pag

if(app_mode=="Home"):
    st.header("Fruits and Vegetable Recognition System")
    image_path="fruits.jpg"
    st.image(image_path)

#About project

elif(app_mode=="About"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:")
    st.text("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.text("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("Train: Contains 100 images per category.")
    st.text("Test: Contains 10 images per category.")
    st.text("Validation: Contains 10 images per category")

#prediction page

elif(app_mode=="Prediction"):
    st.header("Model prediction")
    test_image=st.file_uploader("Choose an image")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        #Reading Labels
        with open('labels.txt') as f:
            content=f.readlines()
        label=[]
        # st.write(content) 
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting it's a {}".format(label[result_index]))


           

