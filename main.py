from flask import Flask, request, render_template
import nltk
import os
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models
from scipy import spatial
from flask_dropzone import Dropzone




nltk.download('stopwords')
nltk.download('punkt')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("english"))
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
embeddings_dict = {}
#with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    #for line in f:
        #values = line.split()
        #word = values[0]
        #vector = np.asarray(values[1:], "float32")
        #embeddings_dict[word] = vector


#def find_closest_embeddings(embedding):
    #return sorted(embeddings_dict.keys(),
                  #key=lambda word_: spatial.distance.euclidean(embeddings_dict[word_], embedding))

#The above code was used to find the topic name. But it was taking a lot of time.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads' #configure path for saving the input file temporarily



# app.config['MAX_CONTENT_PATH']=50*1024*1024
def topics(filename, num):
    with open('./uploads/' + filename, 'r') as file:
        text = file.read().replace('\n', '')
        text1 = text
        
    text = text.lower() #convert text to lower case
    # remove punctuations

    word_tokens = tokenizer.tokenize(text)
    # print(word_tokens)
    # print()
    # removing the stop words.

    text = [word1 for word1 in word_tokens if word1 not in stop_words]
    # print(text)
    # print()
    # removing everything but nouns

    text = nltk.pos_tag(text)
    text = [word1 for word1, tag in text if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS']
    # print(text)

    # lemmatizing the words

    text = [lemmatizer.lemmatize(word1, pos='n') for word1 in text]
    # print(text)
    # print()

    # creating dictionary with unique id

    dictionary = corpora.Dictionary([text])
    # print(dictionary.token2id)

    corpus = [dictionary.doc2bow(text)]
    # print(corpus)

#applying the lda model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num, id2word=dictionary, passes=10,alpha='auto')


    words1 = []
    prob= []


    for i, topic in ldamodel.show_topics(formatted=True, num_topics=num, num_words=3):
        res = tokenizer.tokenize(topic)
        # print(res)
        stop = ['0']
        res = [word for word in res if word not in stop]
        # print(res)
        words = [res[i] for i in range(len(res)) if i % 2 != 0]
        probs = [(int(res[i]) / 1000) for i in range(len(res)) if i % 2 != 1]
        words1.append(words)
        prob.append(probs)

        se = text1.replace('.', '')
        print(se)
        list1 = []
        list2=[]
        list3=[]
        list4=[]

        for sent in nltk.sent_tokenize(se):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    temp = ''
                    for c in chunk:
                        temp += c[0]
                    list1.append((chunk.label(), temp))
        list1=list(dict.fromkeys(list1))

        for e, n in list1: 
            if (e == 'GPE'):  #list2 stores entities with GPE
                list2.append(n)
            elif (e == 'PERSON'):  #list3 stores entities with Person
                list3.append(n)
            else:
                list4.append(n)  #list4 stores entities with Organization


                    # print(chunk.label(), ' '.join(c[0] for c in chunk))

    return words1,prob, list3 , list2,list4


@app.route('/')
def index():
    return render_template('form1.html')


@app.route('/uploader', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # number = request.form['text']
    words,prob,person,gpe,organization = topics(filename, 5)

    return render_template('out.html',words=words,prob=prob,person=person,gpe=gpe,organization=organization)




if __name__ == "__main__":
    app.run(threaded=True, debug=True)
