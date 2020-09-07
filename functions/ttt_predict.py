def predict_tt(iX):
    from keras.models import model_from_json

    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    # ------------------------tic tac toe model
    # load json and create model
    # json_file = open('../d5mit_flask/static/modelAgentLinki.json', 'r')
    json_file = open('static/modelAgentLinki.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_modelL = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_modelL.load_weights("static/modelAgentLinki.h5")
    print("Loaded Agent Linki from disk")
    # evaluate loaded model on test data
    loaded_modelL.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ynew = loaded_modelL.predict_proba(iX)

    return ynew

def printtest():
    print('test')