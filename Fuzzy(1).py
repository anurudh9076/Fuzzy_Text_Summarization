from tkinter import *
from tkinter import filedialog
from rouge import Rouge 
import os
import numpy as np
#import rake_nltk




root = Tk()
my_filetypes = [('all files', '.*'), ('text files', '.txt')]
f_value_arr=[]
# root.filename1 =  filedialog.askopenfilename(title="D:\summary.txt",filetypes=my_filetypes)
file_list=['001.txt','002.txt','003.txt','004.txt','005.txt','006.txt','007.txt','008.txt','009.txt','010.txt']
for i in range(11,25,1):
    if(i!=20):
        file_list.append("0"+str(i)+".txt")


for file_last_name in file_list:
    file_name=r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\article\business'+'\\'+file_last_name
    print(file_name)
 
    # print ("File directory:",root.filename1)
    print("\n")
    root.withdraw()
    #text=open(root.filename, encoding="utf-8").read()
    text=open(file_name).read()
    # print(text)
    print("\n")



    # # Sentence segmentation
    
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

    import nltk
    from string import punctuation
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
    # print("SentenceVectors:",VSM)
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
    #print(keywords)
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
    from nltk.tag import pos_tag
    from collections import Counter
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


    # # feature matrix1


    featureMatrix = []
    featureMatrix.append(position_rbm)
    featureMatrix.append(bi_token_length)
    featureMatrix.append(tri_token_length)
    featureMatrix.append(tfisfvec_rbm)
    featureMatrix.append(cosine_similarity_rbm)
    featureMatrix.append(thematic_number_rbm)
    featureMatrix.append(sent_length_rbm)
    featureMatrix.append(numeric_token_rbm)
    featureMatrix.append(pnounscore_rbm)



    featureMat = np.zeros((len(sentences),9))
    for i in range(9) :
        for j in range(len(sentences)):
            featureMat[j][i] = featureMatrix[i][j]

    # print("\n\n\nPrinting Feature Matrix : ")
    # print(featureMat)
    # print("\n\n\nPrinting Feature Matrix Normed : ")
    #featureMat_normed = featureMat / featureMat.max(axis=0)
    featureMat_normed = featureMat

    # print(featureMat_normed)
    # for i in range(len(sentences)):
    #     print(featureMat_normed[i])
    #np.save('output_labels_10.npy',featureMat_normed)



    import numpy as np
    import matplotlib
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

    

    rule_list=[rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,
    rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,
    rule18,rule19,rule20,rule21,rule21,rule22,rule23,rule24,rule25]
    sent_ctrl = ctrl.ControlSystem(rule_list)
    Sent = ctrl.ControlSystemSimulation(sent_ctrl)
    fuzzemptyarr= np.empty((20,1,2), dtype=object)
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
        if Sent.output['senten'] > 50:
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
    print(file_last_name)
    print("text is ================================================>\n")
    print(text)

    print("summary is ================================================>\n")
    print(res_summary)
    print(len(res_summary))
    hyp_path=r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\summaries\Hypothesis\business'+r'\\'+file_last_name
    ref_path=r'C:\Users\Hp\Desktop\Major_Fuzzy\data\bbc_news_data\summaries\reference\001.txt'

    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    print(scores)

    import json
   
# declaringa a class
    class obj:
        
        # constructor
        def __init__(self, dict1):
            self.__dict__.update(dict1)
    
    def dict2obj(dict1):
        
        # using json.loads method and passing json.dumps
        # method and custom object hook as arguments
        return json.loads(json.dumps(dict1), object_hook=obj)
    obj1 = dict2obj(scores)

    # print(type(scores))
    # print(obj1.rouge-1.f)
    print(type(obj1))
    # print(getattr(obj1))
    rouge_1=getattr(obj1,'rouge-1')
    f_value=getattr(rouge_1,'f')
    f_value_arr.append(f_value)
    print(f_value)
    
    print("\n")

average = sum(f_value_arr)/len(f_value_arr)

print("Average of list: ", round(average,5))


