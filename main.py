from flask import Flask, request, render_template
from keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
import re
import numpy as np
import random
import pandas as pd

app = Flask(__name__)

# -------------------------sentiment model
global graph
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

global questionsGlobal

#-get Questions---------------------------------------------------------------------
def getQuestions(iFileName):

    # create df with questions, input form excel
    questionsDf = pd.read_excel(iFileName)

    return questionsDf


def getQuestion():
    # calculate % of questions left

    nrOfQuest = np.sum(app.questionsGlobal[:, 2])
    if nrOfQuest != 0:
        questLeft = app.questionsGlobal[:, 2]/nrOfQuest
        # get question index and put in questionIndex
        xNew = np.random.multinomial(1, list(questLeft))
        questionIndex = np.argmax(xNew)

        # return question, questionindex, answr
        return app.questionsGlobal[questionIndex, 0], questionIndex, app.questionsGlobal[questionIndex, 1], nrOfQuest

    else:
        # return question, questionindex, answr
        return '', '', '', 0

# questions for KIDS
iFileName = 'static/vrae.xlsx'

questionsDf = getQuestions(iFileName)
app.questionsGlobal = np.array(questionsDf)


# sentement analysis is the home page ---------------------------------------------
@app.route('/')
def my_form():
    return render_template('my-form.html')


# sentement analysis post ---------------------------------------------
@app.route('/', methods=['POST'])
def my_form_post():
    tb._SYMBOLIC_SCOPE.value = True         # resolved error

    itext = request.form['text']

    words = re.sub("[^\w]", " ",  itext).split()

    INDEX_FROM=3   # word index offset

    # import os, ssl
    # if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    #         getattr(ssl, '_create_unverified_context',
    #                 None)): ssl._create_default_https_context = ssl._create_unverified_contex

    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}

    x_ireview = [[word_to_id.get(i, 2) for i in words]]

    tokenizer = Tokenizer(num_words=1000)

    x_predict = tokenizer.sequences_to_matrix(x_ireview, mode='binary')

    ynew =loaded_model.predict_proba(x_predict)

    #
    if ynew[0, 1] < 0.5:
        isentiment = ':('
    else:
        isentiment = ':)'

    # return ynew, prednr
    processed_text = ynew
    # return processed_text
    return render_template('my-form.html', nnoutcome=processed_text, isentiment=isentiment, itext=itext)

# tic tac toe home page ---------------------------------------------
@app.route('/tictactoe')
def my_tictactoe():

    # print('tictactoe')

    p1 = '\xa0'
    p2 = '\xa0'
    p3 = '\xa0'
    p4 = '\xa0'
    p5 = '\xa0'
    p6 = '\xa0'
    p7 = '\xa0'
    p8 = '\xa0'
    p9 = '\xa0'
    iPlay = 'O'
    agent = '2'

    move, x, y = makePrediction(p1, p2, p3, p4, p5, p6, p7, p8, p9, agent)
    boardstate = x
    nnOutcome = y
    iMove = str(move)                               #....
    if spaceIsFree(iMove, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = playMove(iMove, 'X', p1, p2, p3, p4, p5, p6, p7, p8, p9)
        # determine winner/tie or continue game
        if isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, 'X'):  # O Winner
            iEndGame = 'O won!'
        run = False
    # else:
    # print('Space is occupied!')

    strboardstate = str(boardstate)
    strnnOutcome1 = str(nnOutcome[0])
    strnnOutcome2 = str(nnOutcome[1])
    strnnOutcome3 = str(nnOutcome[2])
    strnnOutcome4 = str(nnOutcome[3])
    strnnOutcome5 = str(nnOutcome[4])
    strnnOutcome6 = str(nnOutcome[5])
    strnnOutcome7 = str(nnOutcome[6])
    strnnOutcome8 = str(nnOutcome[7])
    strnnOutcome9 = str(nnOutcome[8])

    return render_template('my-tictactoe.html',
            boardstate=boardstate,
            nnOutcome=nnOutcome,
            strboardstate=strboardstate,
            strnnOutcome1=strnnOutcome1,
            strnnOutcome2=strnnOutcome2,
            strnnOutcome3=strnnOutcome3,
            strnnOutcome4=strnnOutcome4,
            strnnOutcome5=strnnOutcome5,
            strnnOutcome6=strnnOutcome6,
            strnnOutcome7=strnnOutcome7,
            strnnOutcome8=strnnOutcome8,
            strnnOutcome9=strnnOutcome9,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            p6=p6,
            p7=p7,
            p8=p8,
            p9=p9,
            iPlay=iPlay,
            agent=agent)



# card game home page ---------------------------------------------
@app.route('/tictactoecard')
def my_tictactoecard():
    p1 = '\xa0'
    p2 = '\xa0'
    p3 = '\xa0'
    p4 = '\xa0'
    p5 = '\xa0'
    p6 = '\xa0'
    p7 = '\xa0'
    p8 = '\xa0'
    p9 = '\xa0'
    iPlay = 'X'
    return render_template('my-tictactoecard.html',
                           p1=p1,
                           p2=p2,
                           p3=p3,
                           p4=p4,
                           p5=p5,
                           p6=p6,
                           p7=p7,
                           p8=p8,
                           p9=p9,
                           iPlay=iPlay)


# Practice ---------------------------------------------
@app.route('/Practice')
def my_Practice():
    iGrade       = request.args.get('iGrade')
    iWeek        = request.args.get('iWeek')
    iDay         = request.args.get('iDay')
    nrOfQuestAsk = request.args.get('nrOfQuestAsk')
    nrOfQuestCor = request.args.get('nrOfQuestCor')

    if iWeek is None:
        iWeek = '1'
    if iDay is None:
        iDay = '1'
    if nrOfQuestAsk is None:
        nrOfQuestAsk = '0'
    if nrOfQuestCor is None:
        nrOfQuestCor = '0'

    # get questions from DF
    df_week = questionsDf[questionsDf['Week'] == int(iWeek)]
    df_day = df_week[df_week['Dag'] == int(iDay)]
    # store questions and answers in numpy array
    app.questionsGlobal = np.array(df_day)

    question, questionIndex, questionAnswer, nrOfQuest = getQuestion()
    # iCorrect = 'X'

    return render_template('my-Practice.html',
                           iGrade=iGrade,
                           iWeek=iWeek,
                           iDay=iDay,
                           question=question,
                           questionIndex=questionIndex,
                           questionAnswer=questionAnswer,
                           nrOfQuest=nrOfQuest,
                           nrOfQuestAsk=nrOfQuestAsk,
                           nrOfQuestCor=nrOfQuestCor)

def playMove(iMove, iPlay, p1, p2, p3, p4, p5, p6, p7, p8, p9):

    if iMove == '1':
        p1 = iPlay
    if iMove == '2':
        p2 = iPlay
    if iMove == '3':
        p3 = iPlay
    if iMove == '4':
        p4 = iPlay
    if iMove == '5':
        p5 = iPlay
    if iMove == '6':
        p6 = iPlay
    if iMove == '7':
        p7 = iPlay
    if iMove == '8':
        p8 = iPlay
    if iMove == '9':
        p9 = iPlay

    return p1, p2, p3, p4, p5, p6, p7, p8, p9


def spaceIsFree(move, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    iReturn = False
    # print(p1)
    if move == '1' and p1 != 'X' and p1 != 'O':
        iReturn = True
    if move == '2' and p2 != 'X' and p2 != 'O':
        iReturn = True
    if move == '3' and p3 != 'X' and p3 != 'O':
        iReturn = True
    if move == '4' and p4 != 'X' and p4 != 'O':
        iReturn = True
    if move == '5' and p5 != 'X' and p5 != 'O':
        iReturn = True
    if move == '6' and p6 != 'X' and p6 != 'O':
        iReturn = True
    if move == '7' and p7 != 'X' and p7 != 'O':
        iReturn = True
    if move == '8' and p8 != 'X' and p8 != 'O':
        iReturn = True
    if move == '9' and p9 != 'X' and p9 != 'O':
        iReturn = True

    return iReturn


def boardNotFull(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    isfull = True
    if (p1 != 'X' and p1 != 'O'):
        isfull = False
    if (p2 != 'X' and p2 != 'O'):
        isfull = False
    if (p3 != 'X' and p3 != 'O'):
        isfull = False
    if (p4 != 'X' and p4 != 'O'):
        isfull = False
    if (p5 != 'X' and p5 != 'O'):
        isfull = False
    if (p6 != 'X' and p6 != 'O'):
        isfull = False
    if (p7 != 'X' and p7 != 'O'):
        isfull = False
    if (p8 != 'X' and p8 != 'O'):
        isfull = False
    if (p9 != 'X' and p9 != 'O'):
        isfull = False

    # print(isfull)
    if not isfull:
        boardNotFull = True
    else:
        boardNotFull = False

    return boardNotFull


# Check if input is winner
def isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, le):
    return ((p7 == le and p8 == le and p9 == le) or
            (p4 == le and p5 == le and p6 == le) or
            (p1 == le and p2 == le and p3 == le) or
            (p1 == le and p4 == le and p7 == le) or
            (p2 == le and p5 == le and p8 == le) or
            (p3 == le and p6 == le and p9 == le) or
            (p1 == le and p5 == le and p9 == le) or
            (p3 == le and p5 == le and p7 == le))


def iconv(idata, input):
    ret = 0
    if idata == input:
        ret = 1

    return int(ret)


# board to X
def boardToX(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    iX = np.zeros(18, dtype=int)
    iX[0] = iconv(p1, 'X')
    iX[1] = iconv(p2, 'X')
    iX[2] = iconv(p3, 'X')
    iX[3] = iconv(p4, 'X')
    iX[4] = iconv(p5, 'X')
    iX[5] = iconv(p6, 'X')
    iX[6] = iconv(p7, 'X')
    iX[7] = iconv(p8, 'X')
    iX[8] = iconv(p9, 'X')
    iX[9] = iconv(p1, 'O')
    iX[10] = iconv(p2, 'O')
    iX[11] = iconv(p3, 'O')
    iX[12] = iconv(p4, 'O')
    iX[13] = iconv(p5, 'O')
    iX[14] = iconv(p6, 'O')
    iX[15] = iconv(p7, 'O')
    iX[16] = iconv(p8, 'O')
    iX[17] = iconv(p9, 'O')

    return iX


def makePrediction(p1, p2, p3, p4, p5, p6, p7, p8, p9, mode):

    prednr = 0
    iX = np.zeros((1, 18), dtype=int)
    iX[0] = np.array(boardToX(p1, p2, p3, p4, p5, p6, p7, p8, p9))

    #ynew = model.predict_proba(iX)
    if mode == '1':
        # print('1')
        prednr = random.randint(1, 9)
        ynew = np.zeros(9)
        ynew[prednr-1] = 1

    elif mode == '2':
        # print('2')

        ynew = loaded_modelL.predict_proba(iX)
        if np.sum(iX[0]) == 0:
            ynew = np.random.multinomial(1, ynew[0])
        prednr = np.argmax(ynew) + 1
        ynew = np.around(ynew[0:9], decimals=3)
        if ynew.shape[0] == 9:
            ynew = ynew
        else:
            ynew = ynew[0]

        # print(str(ynew))

    elif mode == '3':
        print('3')
    # todo load the model... 111
    #     ynew = loaded_modelL.predict_proba(iX)
    #     if np.sum(iX[0]) == 0:
    #         ynew = np.random.multinomial(1, ynew[0])
    #     prednr = np.argmax(ynew) + 1
    #     ynew = np.around(ynew, decimals=3)

    return prednr, iX[0], ynew


# tic tac toe post code ---------------------------------------------
@app.route('/tictactoe', methods=['POST'])
def my_tictactoe_post():

    agent = request.form['agent']
    iMove = request.form['iMove']
    iPlay = request.form['iPlay']
    p1 = request.form['p1']
    p2 = request.form['p2']
    p3 = request.form['p3']
    p4 = request.form['p4']
    p5 = request.form['p5']
    p6 = request.form['p6']
    p7 = request.form['p7']
    p8 = request.form['p8']
    p9 = request.form['p9']

    nnOutcome = ''
    iEndGame = ''
    boardstate = ''


    # X play
    if spaceIsFree(iMove, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = playMove(iMove, 'O', p1, p2, p3, p4, p5, p6, p7, p8, p9)

        # determine winner/tie or continue game
        if isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, 'O'):  # O Winner
            iEndGame = 'O won!'

        # O play
        run = boardNotFull(p1, p2, p3, p4, p5, p6, p7, p8, p9)
        if iEndGame != '':
            run = False
        while run:
            move, x, y = makePrediction(p1, p2, p3, p4, p5, p6, p7, p8, p9, agent)
            boardstate = x
            nnOutcome = y
            iMove = str(move)
            # print(move)
            if spaceIsFree(iMove, p1, p2, p3, p4, p5, p6, p7, p8, p9):
                p1, p2, p3, p4, p5, p6, p7, p8, p9 = playMove(iMove, 'X', p1, p2, p3, p4, p5, p6, p7, p8, p9)
                # determine winner/tie or continue game
                if isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, 'X'):  # O Winner
                    iEndGame = 'X won!'
                run = False
            # else:
                # print('Space is occupied!')

    strboardstate = str(boardstate)
    strnnOutcome1 = str(nnOutcome[0])
    strnnOutcome2 = str(nnOutcome[1])
    strnnOutcome3 = str(nnOutcome[2])
    strnnOutcome4 = str(nnOutcome[3])
    strnnOutcome5 = str(nnOutcome[4])
    strnnOutcome6 = str(nnOutcome[5])
    strnnOutcome7 = str(nnOutcome[6])
    strnnOutcome8 = str(nnOutcome[7])
    strnnOutcome9 = str(nnOutcome[8])

    return render_template('my-tictactoe.html',
            boardstate=boardstate,
            nnOutcome=nnOutcome,
            strboardstate=strboardstate,
            strnnOutcome1=strnnOutcome1,
            strnnOutcome2=strnnOutcome2,
            strnnOutcome3=strnnOutcome3,
            strnnOutcome4=strnnOutcome4,
            strnnOutcome5=strnnOutcome5,
            strnnOutcome6=strnnOutcome6,
            strnnOutcome7=strnnOutcome7,
            strnnOutcome8=strnnOutcome8,
            strnnOutcome9=strnnOutcome9,
            iEndGame=iEndGame,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            p6=p6,
            p7=p7,
            p8=p8,
            p9=p9,
            iPlay=iPlay,
            agent=agent)


# card game post code ---------------------------------------------
@app.route('/tictactoecard', methods=['POST'])
def my_tictactoecard_post():

    agent = request.form['agent']
    iMove = request.form['iMove']
    iPlay = request.form['iPlay']
    p1 = request.form['p1']
    p2 = request.form['p2']
    p3 = request.form['p3']
    p4 = request.form['p4']
    p5 = request.form['p5']
    p6 = request.form['p6']
    p7 = request.form['p7']
    p8 = request.form['p8']
    p9 = request.form['p9']

    nnOutcome = ''
    iEndGame = ''
    boardstate = ''


    # X play
    if spaceIsFree(iMove, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = playMove(iMove, 'X', p1, p2, p3, p4, p5, p6, p7, p8, p9)

        # determine winner/tie or continue game
        if isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, 'X'):  # O Winner
            iEndGame = 'X won!'

        # O play
        run = boardNotFull(p1, p2, p3, p4, p5, p6, p7, p8, p9)
        if iEndGame != '':
            run = False
        icount = 0
        while run:
            icount = icount + 1
            move, x, y = makePrediction(p1, p2, p3, p4, p5, p6, p7, p8, p9, agent)
            boardstate = x
            nnOutcome = y
            iMove = str(move)
            # print(move)
            if spaceIsFree(iMove, p1, p2, p3, p4, p5, p6, p7, p8, p9):
                p1, p2, p3, p4, p5, p6, p7, p8, p9 = playMove(iMove, 'O', p1, p2, p3, p4, p5, p6, p7, p8, p9)
                # determine winner/tie or continue game
                if isWinner(p1, p2, p3, p4, p5, p6, p7, p8, p9, 'O'):  # O Winner
                    iEndGame = 'O won!'
                run = False
            if icount > 50:
                run = False
            # else:
                # print('Space is occupied!')

    strboardstate = str(boardstate)

    return render_template('my-tictactoecard.html',
            boardstate=boardstate,
            nnOutcome=nnOutcome,
            strboardstate=strboardstate,
            iEndGame=iEndGame,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            p6=p6,
            p7=p7,
            p8=p8,
            p9=p9,
            iPlay=iPlay,
            agent=agent)



# practicve post ---------------------------------------------
@app.route('/Practice', methods=['POST'])
def my_PracticePost():

    iGrade = request.form['iGrade']
    iWeek = request.form['iWeek']
    iDay = request.form['iDay']
    questionIndex = request.form['questionIndex']
    iCorrect = request.form['iCorrect']
    nrOfQuestAsk = request.form['nrOfQuestAsk']
    nrOfQuestCor = request.form['nrOfQuestCor']

    nrOfQuestAsk = int(nrOfQuestAsk) + 1

    if iCorrect == 'X':
        nrOfQuestCor = int(nrOfQuestCor) + 1
        app.questionsGlobal[int(questionIndex), 2] = app.questionsGlobal[int(questionIndex), 2] - 1

    question, questionIndex, questionAnswer, nrOfQuest = getQuestion()

    return render_template('my-Practice.html',
                            iGrade=iGrade,
                            iWeek=iWeek,
                            iDay=iDay,
                            question=question,
                            questionIndex=questionIndex,
                            questionAnswer=questionAnswer,
                            nrOfQuest=nrOfQuest,
                            nrOfQuestAsk=nrOfQuestAsk,
                            nrOfQuestCor=nrOfQuestCor)

    # return render_template('Practice.html')

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80, threaded=False)
    app.run()