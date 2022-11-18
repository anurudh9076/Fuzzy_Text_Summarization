# Core Packages
import os
import tkinter as tk
import tkinter.filedialog
from tkinter import *
from tkinter import filedialog, ttk
from tkinter.scrolledtext import *

from rouge import Rouge

#import rake_nltk




 # Structure and Layout
window = Tk()
window.title("Summaryzer GUI")
window.geometry("780x400")
window.config(background='black')

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn',)


# TAB LAYOUT
tab_control = ttk.Notebook(window,style='lefttab.TNotebook')
 
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Home":^20s}')
tab_control.add(tab2, text=f'{"File":^20s}')
tab_control.add(tab3, text=f'{"URL":^20s}')
tab_control.add(tab4, text=f'{"About ":^20s}')


label1 = Label(tab1, text= 'Summaryzer',padx=5, pady=5)
label1.grid(column=0, row=0)
 
label2 = Label(tab2, text= 'File Processing',padx=5, pady=5)
label2.grid(column=0, row=0)

label3 = Label(tab3, text= 'URL',padx=5, pady=5)
label3.grid(column=0, row=0)


label4 = Label(tab4, text= 'About',padx=5, pady=5)
label4.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')


def getSummary(text):
    
    
    import numpy as np
    from nltk import sent_tokenize
    sentences=(sent_tokenize(text))
    # print("Sentences:",sentences)
    # print("\n")
    #print(len(sentences))


    emptyarray= np.empty((len(sentences),1,3),dtype=object)
    for s in range(len(sentences)):
        emptyarray[s][0][0] = sentences[s]
        emptyarray[s][0][1] = s


    # # Tokenization, Stop word removal , Bi-grams, Tri-grams

    from string import punctuation

    import nltk
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    bi_token=[]
    bi_token_length=[]
    tri_token_length=[]
    for u in range(len(sentences)):
        sent_split1=[w.lower() for w in sentences[u].split(" ")]
        sent_split=[w for w in sent_split1 if w not in stop_words and w not in punctuation and not w.isdigit()]
        
        bigrams_list = [bigram for bigram in nltk.bigrams(sent_split)]
        bi_token.append(bigrams_list)
        bi_token_length.append(len(bi_token[u]))
    bi_tokens = [(int(o) / max(bi_token_length))*100 for o in bi_token_length]
    # print("bitokens feature vector:",(bi_token_length))
    #print(max(bi_token_length))
    #print(bi_token_length)
    print("\n")


    tri_token=[]
    for u in range(len(sentences)):
        sent_split2=[w.lower() for w in sentences[u].split(" ")]
        sent_split3=[w for w in sent_split2 if w not in stop_words and w not in punctuation and not w.isdigit()]
        trigrams_list = [trigram for trigram in nltk.trigrams(sent_split3)]
        tri_token.append(trigrams_list)
        tri_token_length.append(len(tri_token[u]))
    tri_tokens = [(int(m) / max(tri_token_length))*100 for m in tri_token_length]

    # print("tritokens feature vector:",tri_token_length)
    # print("\n")



    # # Sentence Position Feature

    import math
    def position(l):
        return [index for index, value in enumerate(sentences)]

    sent_position= (position(sentences))
    num_sent=len(sent_position)
    # print("sentence position:",sent_position)
    # print("\n")
    # print("Total number of sentences:",num_sent)
    # print("\n")
    #th= 0.2*num_sent
    #minv=th*num_sent
    #maxv=th*2*num_sent
    position = []
    position_rbm = []
    sent_pos1_rbm = 1
    sent_pos1 = 100
    position.append(sent_pos1)
    position_rbm.append(sent_pos1_rbm)
    for x in range(1,num_sent-1):
        #s_p = (math.cos((sent_position[x]-minv)*((1/maxv)-minv)))*100
        #if s_p < 0:
        #   s_p = 0
        s_p= ((num_sent-x)/num_sent)*100
        position.append(s_p)
        s_p_rbm = (num_sent-x)/num_sent
        position_rbm.append(s_p_rbm)
        
    sent_pos2 = 100
    sent_pos2_rbm = 1
    position.append(sent_pos2)
    position_rbm.append(sent_pos2_rbm)
    # print("Sentence position feature vector:",position_rbm)
    # print("\n")



    # # Converting Sentences to Vectors

    def convertToVSM(sentences):
        vocabulary = []
        for sents in sentences:
            vocabulary.extend(sents)
        vocabulary = list(set(vocabulary))
        vectors = []
        for sents in sentences:
            vector = []
            for tokenss in vocabulary:
                vector.append(sents.count(tokenss))
            vectors.append(vector)
        return vectors
    VSM=convertToVSM(sentences)
    print("SentenceVectors:",VSM)
    # print("\n")




    # # TF-ISF feature and Centroid Calculation

    sentencelength=len(sentences)
    def calcMeanTF_ISF(VSM, index):
        vocab_len = len(VSM[index])
        sentences_len = len(VSM)
        count = 0
        tfisf = 0
        for i in range(vocab_len):
            tf = VSM[index][i]
            if(tf>0):
                count += 1
                sent_freq = 0
                for j in range(sentences_len):
                    if(VSM[j][i]>0): sent_freq += 1
                tfisf += (tf)*(1.0/sent_freq)
        if(count > 0):
            mean_tfisf = tfisf/count
        else:
            mean_tfisf = 0
        return tf, (1.0/sent_freq), mean_tfisf
    tfvec=[]
    isfvec=[]
    tfisfvec=[]
    tfisfvec_rbm=[]
    for i in range(sentencelength):
        x,y,z=calcMeanTF_ISF(VSM,i)
        tfvec.append(x)
        isfvec.append(y)
        tfisfvec.append(z*100)
        tfisfvec_rbm.append(z)
    #print("TF vector:",tfvec)
    #print("\n")
    #print("ISF vector:",isfvec)
    #print("\n")
    #tfisf1= [(int(p)*100) for p in tfisfvec]
    # print("TF-ISF vector:",tfisfvec_rbm)
    # print("\n")
    maxtf_isf=max(tfisfvec_rbm)
    centroid=[]
    centroid.append(maxtf_isf)
    # print("Max TF-ISF:",centroid)
    # print("\n")
    #for q in range(sentencelength):
    centroid=(max(VSM))
    # print("Centroid:",centroid)
    # print("\n")



    # # Cosine Similarity between Centroid and Sentences

    from numpy import dot
    from numpy.linalg import norm
    cosine_similarity=[]
    cosine_similarity_rbm=[]
    for z in range(sentencelength):
        cos_simi = ((dot(centroid, VSM[z])/(norm(centroid)*norm(VSM[z])))*100)
        cosine_similarity.append(cos_simi)
        cos_simi_rbm = (dot(centroid, VSM[z])/(norm(centroid)*norm(VSM[z])))
        cosine_similarity_rbm.append(cos_simi_rbm)
    # print("Cosine Similarity Vector:",cosine_similarity_rbm)
    # print("\n")




    # # Sentence length feature

    sent_word=[]
    for u in range(len(sentences)):
        sent_split1=[w.lower() for w in sentences[u].split(" ")]
        sent_split=[w for w in sent_split1 if w not in stop_words and w not in punctuation and not w.isdigit()]
        a=(len(sent_split))
        sent_word.append(a)
    #print("Number of words in each sentence:",sent_word)
    #print("\n")
    #sent_leng=[]
    #for x in range(len(sentences)):
    #   if sent_word[x] < 3:
    #      sent_leng.append(0)
    #  else:
    #     sent_leng.append(1)

    ##OR BY THIS METHOD: LENGTH OF SENTENCE/ LONGEST SENTENCE
    longest_sent=max(sent_word)
    sent_length=[]
    sent_length_rbm=[]
    for x in sent_word:
        sent_length.append((x/longest_sent)*100)
        sent_length_rbm.append(x/longest_sent)
    #print(sent_length)

    # print("Sentence length feature vector:",sent_length_rbm)
    # print("\n")




    # # Numeric token Feature

    import re
    num_word=[]
    numeric_token=[]
    numeric_token_rbm=[]
    for u in range(len(sentences)):
        sent_split4=sentences[u].split(" ")
        e=re.findall("\d+",sentences[u])
        noofwords=(len(e))
        num_word.append(noofwords)
        numeric_token.append((num_word[u]/sent_word[u])*100)
        numeric_token_rbm.append(num_word[u]/sent_word[u])
    #print("Numeric word count in each sentence:",num_word)
    #print("\n")
    # print("Numeric token feature vector:",numeric_token_rbm)
    # print("\n")



    # # Thematic words feature

    from rake_nltk import Rake
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    keywords=[]
    # If you want to provide your own set of stop words and punctuations to
    # r = Rake(<list of stopwords>, <string of puntuations to ignore>)

    for s in sentences:
        r.extract_keywords_from_text(s)
        key=list(r.get_ranked_phrases())
        keywords.append(key)
    # print(keywords)
    l_keywords=[]
    for s in keywords:
        leng=len(s)
        l_keywords.append(leng)
    #print(l_keywords)

    total_keywords=sum(l_keywords)
    #print(total_keywords)

    thematic_number= []
    thematic_number_rbm= []
    for x in l_keywords:
        thematic_number.append((x/total_keywords)*100)
        thematic_number_rbm.append(x/total_keywords)
    # print("Thematic word feature", thematic_number_rbm)
    # print("\n")



    # # proper noun feature
    from collections import Counter

    from nltk.tag import pos_tag
    pncounts = []
    pncounts_rbm = []
    for sentence in sentences:
        tagged=nltk.pos_tag(nltk.word_tokenize(str(sentence)))
        counts = Counter(tag for word,tag in tagged if tag.startswith('NNP') or tag.startswith('NNPS'))
        f=sum(counts.values())
        pncounts.append(f)
        pncounts_rbm.append(f)
    pnounscore=[(int(o) / int(p))*100 for o,p in zip(pncounts, sent_word)]
    pnounscore_rbm=[int(o) / int(p) for o,p in zip(pncounts_rbm, sent_word)]
    # #print(pncounts)
    # print("Pronoun feature vector",pnounscore_rbm)
    # print("\n")






    import matplotlib
    import numpy as np
    #%matplotlib inline
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions
    position1 = ctrl.Antecedent(np.arange(0, 100, 10), 'position1')
    cos_similarity = ctrl.Antecedent(np.arange(0, 100, 10), 'cos_similarity')
    bitokens = ctrl.Antecedent(np.arange(0, 100, 10), 'bitokens')
    tritokens = ctrl.Antecedent(np.arange(0, 100, 10), 'tritokens')
    propernoun = ctrl.Antecedent(np.arange(0, 100, 10), 'propernoun')
    sentencelength = ctrl.Antecedent(np.arange(0, 100, 10), 'sentencelength')
    numtokens = ctrl.Antecedent(np.arange(0, 100, 10), 'numtokens')
    keywords = ctrl.Antecedent(np.arange(0, 10, 1), 'keywords')
    tf_isf = ctrl.Antecedent(np.arange(0, 100, 10), 'tf_isf')



    senten = ctrl.Consequent(np.arange(0, 100, 10), 'senten')

    position1.automf(3)
    cos_similarity.automf(3)
    bitokens.automf(3)
    tritokens.automf(3)
    propernoun.automf(3)
    sentencelength.automf(3)
    numtokens.automf(3)
    keywords.automf(3)
    tf_isf.automf(3)


    senten['bad'] = fuzz.trimf(senten.universe, [0, 0, 50])
    senten['avg'] = fuzz.trimf(senten.universe, [0, 50, 100])
    senten['good'] = fuzz.trimf(senten.universe, [50, 100, 100])
    
    # senten.view()
    

    rule1 = ctrl.Rule(position1['good'] & sentencelength['good'] & propernoun['good'] &numtokens['good'], senten['good'])
    rule2 = ctrl.Rule(position1['poor'] & sentencelength['poor'] & numtokens['poor'], senten['bad'])
    rule3 = ctrl.Rule(propernoun['poor'] & keywords['average'], senten['bad'])
    rule4 = ctrl.Rule(cos_similarity['good'], senten['good'])
    rule5 = ctrl.Rule(bitokens['good'] & tritokens['good'] & numtokens['average'] | tf_isf['average'], senten['avg'])
    rule6 = ctrl.Rule(bitokens['good'] & tritokens['good'] & numtokens['good'] | tf_isf['good'], senten['good'])
    rule7 = ctrl.Rule(bitokens['good'] & tritokens['poor'] & numtokens['poor'] | tf_isf['poor'], senten['bad'])
    rule8 = ctrl.Rule(bitokens['poor'] & tritokens['poor'] & numtokens['good'] | tf_isf['good'], senten['avg'])
    rule9 = ctrl.Rule(bitokens['poor'] & tritokens['poor'] & numtokens['poor'] | tf_isf['good'], senten['bad'])
    rule10 = ctrl.Rule(bitokens['average'] & tritokens['good'] & numtokens['good'] | tf_isf['good'], senten['good'])
    rule11 = ctrl.Rule(bitokens['poor'] & tritokens['poor'] & numtokens['good'] | tf_isf['good'], senten['avg'])
    rule12 = ctrl.Rule(propernoun['average'] & tritokens['poor'] & numtokens['good'] | tf_isf['good'], senten['avg'])
    rule13= ctrl.Rule(position1['good'] & numtokens['good'] | tf_isf['good'], senten['good'])
    rule14= ctrl.Rule(position1['poor'] & numtokens['good'] | tf_isf['good'], senten['avg'])
    rule15= ctrl.Rule(position1['poor'] & numtokens['average'] | tf_isf['good'], senten['avg'])
    rule16= ctrl.Rule(position1['good'] & numtokens['poor'] | tf_isf['average'], senten['avg'])
    rule17= ctrl.Rule(position1['poor'] & numtokens['poor'] | tf_isf['poor'], senten['bad'])
    rule18= ctrl.Rule(position1['good'] & numtokens['poor'] | tf_isf['good'], senten['avg'])
    rule19= ctrl.Rule(position1['good'] & keywords['good'] | tf_isf['good'], senten['good'])
    rule20= ctrl.Rule(position1['good'] & keywords['poor'] | tf_isf['good'], senten['avg'])
    rule21= ctrl.Rule(position1['good'] & keywords['poor'] | tf_isf['poor'], senten['bad'])
    rule22= ctrl.Rule(position1['poor'] & keywords['poor'] | tf_isf['poor'], senten['bad'])
    rule23= ctrl.Rule(position1['good'] & keywords['good'] & cos_similarity['good'] & numtokens['good'] | tf_isf['good'], senten['good'])
    rule24= ctrl.Rule(position1['good'] & keywords['poor'] & cos_similarity['good'] & numtokens['average'] | tf_isf['good'], senten['avg'])
    rule25= ctrl.Rule(position1['poor'] & keywords['poor'] & cos_similarity['poor'] & numtokens['poor'] | tf_isf['poor'], senten['bad'])
    rule26= ctrl.Rule(position1['good'] & propernoun['good'] & keywords['poor'] & cos_similarity['poor'] & numtokens['poor'] | tf_isf['good'], senten['avg'])
    rule27= ctrl.Rule(position1['good'] & propernoun['good'] & keywords['good'] & cos_similarity['poor'] & numtokens['good'] | tf_isf['good'], senten['good'])
    rule28= ctrl.Rule(position1['good'] & propernoun['good'] & keywords['good'] & cos_similarity['good'] & numtokens['poor'] | tf_isf['good'], senten['good'])
    rule29= ctrl.Rule(position1['good'] & propernoun['good'] & keywords['poor'] & cos_similarity['good'] & numtokens['poor'] | tf_isf['average'], senten['avg'])
    rule30= ctrl.Rule(position1['poor'] & propernoun['poor'] & keywords['poor'] & cos_similarity['poor'] & numtokens['poor'] | tf_isf['good'], senten['bad'])
    rule31= ctrl.Rule(propernoun['poor'] & keywords['poor'] & cos_similarity['poor'], senten['bad'])
    rule32= ctrl.Rule(propernoun['good'] & keywords['good'] & cos_similarity['good'], senten['good'])
    rule33= ctrl.Rule(propernoun['good'] & keywords['poor'] & cos_similarity['good'], senten['avg'])
    rule34= ctrl.Rule(propernoun['average'] & keywords['average'] & cos_similarity['average'], senten['avg'])
    
    rule35= ctrl.Rule(propernoun['good'] & sentencelength['good'] & keywords['good'] & cos_similarity['good'], senten['good'])
    rule36= ctrl.Rule(propernoun['average'] & sentencelength['average'] & keywords['average'] & cos_similarity['average'], senten['avg'])
    rule37= ctrl.Rule(propernoun['average'] & sentencelength['average'] & keywords['good'] & cos_similarity['poor'], senten['avg'])
    rule38= ctrl.Rule(propernoun['good'] & sentencelength['average'] & keywords['poor'] & cos_similarity['average'], senten['avg'])
    rule39= ctrl.Rule(propernoun['poor'] & sentencelength['average'] & keywords['good'] & cos_similarity['average'], senten['avg'])
    rule40= ctrl.Rule(propernoun['poor'] & sentencelength['poor'] & keywords['poor'] & cos_similarity['poor'], senten['bad'])



    

    rule_list=[rule1,rule2,rule3,rule4,rule5] #,rule6,rule7,rule8,
    # rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,
    # rule18,rule19,rule20,rule21,rule21,rule22,rule23,rule24,rule25
    sent_ctrl = ctrl.ControlSystem(rule_list)
    Sent = ctrl.ControlSystemSimulation(sent_ctrl)
    summary_len=int(lenInput.get())
    fuzzemptyarr= np.empty((summary_len,1,2), dtype=object)
    t2=0
    

    summary2=[]
    for s in range(len(sentences)):
        Sent.input['position1'] = int(position[s])
        Sent.input['cos_similarity'] = int(cosine_similarity[s])
        Sent.input['bitokens'] = int(bi_tokens[s])
        Sent.input['tritokens'] = int(tri_tokens[s])
        Sent.input['tf_isf'] = int(tfisfvec[s])
        Sent.input['keywords'] = int(thematic_number[s])
        Sent.input['propernoun'] = int(pnounscore[s])
        Sent.input['sentencelength'] = int(sent_length[s])
        Sent.input['numtokens'] = int(numeric_token[s])
    #Sent.input['service'] = 2
        Sent.compute()
        if Sent.output['senten'] > 50 and t2<summary_len:
            summary2.append((sentences[s]))
            fuzzemptyarr[t2][0][0] = sentences[s]
            fuzzemptyarr[t2][0][1] = s
            t2+=1
    fuzzarray = np.empty((len(summary2),1,2),dtype=object)
    for i in range(len(summary2)):
        fuzzarray[i][0][0] = fuzzemptyarr[i][0][0]
        fuzzarray[i][0][1] = fuzzemptyarr[i][0][1]
        
    fuzzarray=fuzzarray[1:]
    # print("Fuzzy logic summary \n\n",summary2)

    # print("before n removal summary is \n")
    # print(summary2)

    res_summary=""

    #  for i in range(len(summary2)):
    #      summary2[i]=summary2[i].strip()


    summary3=[]
    import re
    for i in summary2 :
        summary3.append(re.sub('\n','',i))

    for i in range(len(summary3)):
        res_summary+=summary3[i]

    # print("after n removal summary is \n")
    # print(res_summary)


        # res_summary= res_summary.strip()


    # res_summary=res_summary.replace(' '+"/n"+' ',' ')

    # print(res_summary)




    # root.filename2 =  filedialog.askopenfilename(title="D:\summary.txt",filetypes=my_filetypes)
    # roo/t.withdraw()

    # hypothesis = open(root.filename2).read() 

    # reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"


    # print(res_summary)
    # rouge = Rouge()
    # scores = rouge.get_scores(hypothesis, res_summary)


    from rouge import FilesRouge

    files_rouge = FilesRouge()

    file1 = open(r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\summaries\reference\001.txt', "w")  # write mode
    file1.write(res_summary)
    file1.close()

    # print("success 1")
    # print(x)
    
    print("text is ================================================>\n")
    print(text)

    print("summary is ================================================>\n")
    print(res_summary)
    print(len(res_summary))
    # hyp_path=r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\summaries\Hypothesis\business'+r'\\'+file_last_name
    # ref_path=r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\summaries\reference\001.txt'
    return res_summary

# Functions 
def get_summary():
    text = str(entry.get('1.0',tk.END))
    res_summary=getSummary(text)
    tab1_display.insert(tk.END,res_summary)


# Clear entry widget
def clear_text():
	entry.delete('1.0',END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Clear Text  with position 1.0
def clear_text_file():
	displayed_file.delete('1.0',END)

# Clear Result of Functions
def clear_text_result():
	tab2_display_text.delete('1.0',END)

# Clear For URL
def clear_url_entry():
	url_entry.delete(0,END)

def clear_url_display():
	tab3_display_text.delete('1.0',END)


# # Clear entry widget
# def clear_compare_text():
# 	entry1.delete('1.0',END)

# def clear_compare_display_result():
# 	tab1_display.delete('1.0',END)


# Functions for TAB 2 FILE PROCESSER
# Open File to Read and Process
def openfiles():
	file1 = tkinter.filedialog.askopenfilename(filetypes=(("Text Files",".txt"),("All files","*")))
	read_text = open(file1).read()
	displayed_file.insert(tk.END,read_text)


def get_file_summary():
    # # Sentence segmentation
    text = displayed_file.get('1.0',tk.END)
    res_summary=getSummary(text)

	
	# final_text = text_summarizer(raw_text)
	# result = '\nSummary:{}'.format(final_text)
    tab2_display_text.insert(tk.END,res_summary)

# Fetch Text From Url
from urllib.request import urlopen

from bs4 import BeautifulSoup
def get_text():
	raw_text = str(url_entry.get())
	page = urlopen(raw_text)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	url_display.insert(tk.END,fetched_text)

def get_url_summary():
	raw_text = url_display.get('1.0',tk.END)
	final_text = getSummary(raw_text)
	result = '\nSummary:{}'.format(final_text)
	tab3_display_text.insert(tk.END,result)	


# COMPARER FUNCTIONS

# def use_spacy():
# 	raw_text = str(entry1.get('1.0',tk.END))
# 	final_text = text_summarizer(raw_text)
# 	print(final_text)
# 	result = '\nSpacy Summary:{}\n'.format(final_text)
# 	tab4_display.insert(tk.END,result)

# def use_nltk():
# 	raw_text = str(entry1.get('1.0',tk.END))
# 	final_text = nltk_summarizer(raw_text)
# 	print(final_text)
# 	result = '\nNLTK Summary:{}\n'.format(final_text)
# 	tab4_display.insert(tk.END,result)

# def use_gensim():
# 	raw_text = str(entry1.get('1.0',tk.END))
# 	final_text = summarize(raw_text)
# 	print(final_text)
# 	result = '\nGensim Summary:{}\n'.format(final_text)
# 	tab4_display.insert(tk.END,result)

# def use_sumy():
# 	raw_text = str(entry1.get('1.0',tk.END))
# 	final_text = text_summarizer(raw_text)
# 	print(final_text)
# 	result = '\nSumy Summary:{}\n'.format(final_text)
# 	tab4_display.insert(tk.END,result)

# MAIN NLP TAB
l1=Label(tab1,text="Enter Text To Summarize")
l1.grid(row=1,column=0)

entry=Text(tab1,height=10)
entry.grid(row=2,column=0,columnspan=2,padx=5,pady=5)

# BUTTONS
button1=Button(tab1,text="Reset",command=clear_text, width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab1,text="Summarize",command=get_summary, width=12,bg='#ced',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab1,text="Clear Result", command=clear_display_result,width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab1,text="Main Points", width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)

# Display Screen For Result
tab1_display = Text(tab1)
tab1_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


#FILE PROCESSING TAB
l1=Label(tab2,text="Open File To Summarize")
l1.grid(row=1,column=1)

displayed_file = ScrolledText(tab2,height=7)# Initial was Text(tab2)
displayed_file.grid(row=2,column=0, columnspan=3,padx=5,pady=3)

# BUTTONS FOR SECOND TAB/FILE READING TAB
b0=Button(tab2,text="Open File", width=12,command=openfiles,bg='#c5cae9')
b0.grid(row=3,column=0,padx=10,pady=10)

b1=Button(tab2,text="Reset ", width=12,command=clear_text_file,bg="#b9f6ca")
b1.grid(row=3,column=1,padx=10,pady=10)


  
# creating a entry for input
# name using widget Entry

length_label = tk.Label(tab2, text = 'Summary Length', font=('calibre',10,'bold'))
length_label.grid(row=3,column=2,padx=1,pady=1)
lenInput = tk.Entry(tab2,width=3, font=('calibre',10,'normal'))
lenInput.grid(row=4,column=2)
lenInput.insert(0, "10")

b3=Button(tab2,text="Clear Result", width=12,command=clear_text_result)
b3.grid(row=5,column=0,padx=10,pady=10)

b4=Button(tab2,text="Close", width=12,command=window.destroy)
b4.grid(row=5,column=1,padx=10,pady=10)

b2=Button(tab2,text="Summarize", width=12,command=get_file_summary,bg='blue',fg='#fff')
b2.grid(row=5,column=2,padx=10,pady=10)

# Display Screen
# tab2_display_text = Text(tab2)
tab2_display_text = ScrolledText(tab2,height=10)
tab2_display_text.grid(row=7,column=0, columnspan=3,padx=5,pady=5)

# Allows you to edit
tab2_display_text.config(state=NORMAL)


# URL TAB
l1=Label(tab3,text="Enter URL To Summarize")
l1.grid(row=1,column=0)

raw_entry=StringVar()
url_entry=Entry(tab3,textvariable=raw_entry,width=50)
url_entry.grid(row=1,column=1)

# BUTTONS
button1=Button(tab3,text="Reset",command=clear_url_entry, width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab3,text="Get Text",command=get_text, width=12,bg='#03A9F4',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab3,text="Clear Result", command=clear_url_display,width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab3,text="Summarize",command=get_url_summary, width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)

# Display Screen For Result
url_display = ScrolledText(tab3,height=10)
url_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


tab3_display_text = ScrolledText(tab3,height=10)
tab3_display_text.grid(row=10,column=0, columnspan=3,padx=5,pady=5)


# COMPARER TAB
# l1=Label(tab4,text="Enter Text To Summarize")
# l1.grid(row=1,column=0)

# entry1=ScrolledText(tab4,height=10)
# entry1.grid(row=2,column=0,columnspan=3,padx=5,pady=3)

# BUTTONS
# button1=Button(tab4,text="Reset",command=clear_compare_text, width=12,bg='#03A9F4',fg='#fff')
# button1.grid(row=4,column=0,padx=10,pady=10)

# button2=Button(tab4,text="SpaCy",command=use_spacy, width=12,bg='red',fg='#fff')
# button2.grid(row=4,column=1,padx=10,pady=10)

# button3=Button(tab4,text="Clear Result", command=clear_compare_display_result,width=12,bg='#03A9F4',fg='#fff')
# button3.grid(row=5,column=0,padx=10,pady=10)

# button4=Button(tab4,text="NLTK",command=use_nltk, width=12,bg='#03A9F4',fg='#fff')
# button4.grid(row=4,column=2,padx=10,pady=10)

# button4=Button(tab4,text="Gensim",command=use_gensim, width=12,bg='#03A9F4',fg='#fff')
# button4.grid(row=5,column=1,padx=10,pady=10)

# button4=Button(tab4,text="Sumy",command=use_sumy, width=12,bg='#03A9F4',fg='#fff')
# button4.grid(row=5,column=2,padx=10,pady=10)


# variable = StringVar()
# variable.set("SpaCy")
# choice_button = OptionMenu(tab4,variable,"SpaCy","Gensim","Sumy","NLTK")
# choice_button.grid(row=6,column=1)


# # Display Screen For Result
# tab4_display = ScrolledText(tab4,height=15)
# tab4_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


# About TAB
about_label = Label(tab4,text="Fuzzy Logic Based Text Summmarizer GUI V.0.0.1 \n @Anubhav Rajput \n@Anurudh Pratap Singh ",pady=5,padx=5)
about_label.grid(column=0,row=1)

window.mainloop()


