import plotly.graph_objs as go

def sentiment_predict(x_predict):
#   sentiment model
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


def sentiment_do_check(itext):
    '''
    takes in the text as itext and returns the sentiment analysis.

            Parameters:
                    itext (string): the string that will be analysed

            Returns:
                    ynew (array): array with the probabilities
                    isentiment (string): a :) or a :(
    '''

    # imports
    import re
    import keras.backend.tensorflow_backend as tb
    from keras.datasets import imdb
    from keras.preprocessing.text import Tokenizer

    tb._SYMBOLIC_SCOPE.value = True         # resolved error
    words = re.sub("[^\w]", " ",  itext).split()
    INDEX_FROM = 3   # word index offset


    # get the word index from imbd
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}

    x_ireview = [[word_to_id.get(i, 2) for i in words]]

    tokenizer = Tokenizer(num_words=1000)

    x_predict = tokenizer.sequences_to_matrix(x_ireview, mode='binary')

    # call keras
    ynew = sentiment_predict(x_predict)

    # just some simple classification into :), :| and :(
    if ynew[0, 1] < 0.5:
        if ynew[0, 1] > 0.45:
            isentiment = ':|'
        else:
            isentiment = ':('
    else:
        if ynew[0, 1] < 0.55:
            isentiment = ':|'
        else:
            isentiment = ':)'

    return ynew, isentiment, ynew[0, 0], ynew[0, 1]


def get_figures(iunhappy, ihappy):
    ####
    graph_one = []
    # # df = cleandata('data/API_AG.LND.ARBL.HA.PC_DS2_en_csv_v2.csv')
    # df =
    # df.columns = ['Unhappy','Happy']
    # # df.sort_values('hectaresarablelandperperson', ascending=False, inplace=True)
    # df = df[df['year'] == 2015]

    print(iunhappy)
    print(ihappy)

    graph_one.append(
        go.Bar(
        x = ['Unhappy', 'Happy'],
        # y = ynew,
        y = [iunhappy, ihappy]
        )
    )

    layout_one = dict(title = 'Sentiment Analysis',
                xaxis = dict(title = 'Sentiment',),
                yaxis = dict(title = 'Happiness score', range=[0, 1], dtick=0.2, autorange=False),
                )

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))

    return figures

###

