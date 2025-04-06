def model_prediction(test_image):
    model=tf.keras.models.load_model("C:\\Users\\harsh\\OneDrive\\Desktop\\WEbAPP\\trained_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

