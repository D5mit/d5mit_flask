def sentiment_predict(x_predict):
# -------------------------sentiment model
    from keras.models import model_from_json
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    # json_file = open('../d5mit_flask/static/modelSentiment.json', 'r')
    json_file = open('static/modelSentiment.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("static/modelSentiment.h5")
    # # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('model loaded')
    print(loaded_model)

    ynew = loaded_model.predict_proba(x_predict)
    return ynew