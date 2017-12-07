import pandas as pn
from nltk import word_tokenize
from nltk import pos_tag 
from nltk import ne_chunk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import csv
import sys
import string

reload(sys)
sys.setdefaultencoding('latin-1')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.update(['@','>','<','.',',','-','=','(',')','[',']','/','\\','?','_',
'Hey','How','Does','AND','Has','`','Can','Could','First','Would'])

logs = pn.read_csv('logs.csv', encoding = 'latin-1', engine='python')

#extract column from dataframe
vector = logs_filtered['message_utf8']

tg_list = []
for item in vector.iteritems():
    try:
        tokenized_item = word_tokenize(item[1])
    except TypeError:
        pass
    stemmed_lemmatized_list = []
    for word in tokenized_item:
        lemmatized_item = lemmatizer.lemmatize(word)
        #UNCOMMENT IF STEMMING IS TO BE APPLIED
        #stemmed_item = stemmer.stem(lemmatized_item)
        stemmed_lemmatized_list.append(lemmatized_item)
   
    filtered_item = [items for items in stemmed_lemmatized_list if not items in stop_words]
    pos_item = pos_tag(filtered_item)
    tg_item_tree = ne_chunk(pos_item)
    tree_leaves = tg_item_tree.leaves()
    
    itr1 = 0
    while(itr1 < len(tree_leaves)):
     
       if(tree_leaves[itr1][1] == 'NNP'):
            item = tree_leaves[itr1][0]
            tg_list.append(item)
            
       itr1 += 1

    #UNCOMMENT FOR TESTING ON LIMITED NUMBER OF ITEMS:
    #if(itr == 10):
    #    break
    
most_common = [item for item in Counter(tg_list).most_common()]
print(str(most_common))

#UNCOMMENT WHEN SAVING CSV
#with open('top_terms.csv', 'wb') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(most_common)



