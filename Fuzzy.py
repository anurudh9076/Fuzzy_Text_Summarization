from tkinter import *
from tkinter import filedialog
import os
import numpy as np
import rake_nltk
root = Tk()
my_filetypes = [('all files', '.*'), ('text files', '.txt')]
root.filename =  filedialog.askopenfilename(title="D:\summary.txt",filetypes=my_filetypes)

print ("File directory:",root.filename)
print("\n")
root.withdraw()
#text=open(root.filename, encoding="utf-8").read()
text=open(root.filename).read()
print(text)
print("\n")



# # Sentence segmentation

from nltk import sent_tokenize
sentences=(sent_tokenize(text))
print("Sentences:",sentences)
print("\n")
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
print("bitokens feature vector:",(bi_token_length))
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

print("tritokens feature vector:",tri_token_length)
print("\n")



# # Sentence Position Feature

import math
def position(l):
    return [index for index, value in enumerate(sentences)]

sent_position= (position(sentences))
num_sent=len(sent_position)
print("sentence position:",sent_position)
print("\n")
print("Total number of sentences:",num_sent)
print("\n")
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
print("Sentence position feature vector:",position_rbm)
print("\n")



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
print("\n")




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
print("TF-ISF vector:",tfisfvec_rbm)
print("\n")
maxtf_isf=max(tfisfvec_rbm)
centroid=[]
centroid.append(maxtf_isf)
print("Max TF-ISF:",centroid)
print("\n")
#for q in range(sentencelength):
centroid=(max(VSM))
print("Centroid:",centroid)
print("\n")



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
print("Cosine Similarity Vector:",cosine_similarity_rbm)
print("\n")




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

print("Sentence length feature vector:",sent_length_rbm)
print("\n")




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
print("Numeric token feature vector:",numeric_token_rbm)
print("\n")



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
print("Thematic word feature", thematic_number_rbm)
print("\n")



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
#print(pncounts)
print("Pronoun feature vector",pnounscore_rbm)
print("\n")


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

print("\n\n\nPrinting Feature Matrix : ")
print(featureMat)
print("\n\n\nPrinting Feature Matrix Normed : ")
#featureMat_normed = featureMat / featureMat.max(axis=0)
featureMat_normed = featureMat

print(featureMat_normed)
for i in range(len(sentences)):
    print(featureMat_normed[i])
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
rule6 = ctrl.Rule(position1['good'] & sentencelength['good'] & propernoun['poor'] &numtokens['poor'], senten['avg'])
rule7 = ctrl.Rule(position1['good'] & sentencelength['poor'] & propernoun['poor'] &numtokens['average'], senten['avg'])
rule8 = ctrl.Rule(position1['good'] & sentencelength['good'], senten['good'])




sent_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8])
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
print("Fuzzy logic summary \n\n",summary2)




    
