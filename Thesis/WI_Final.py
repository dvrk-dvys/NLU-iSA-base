from tornado import autoreload
import os
import pylab as plt
import nltk
# from hSBM_Topicmodel import sbmtm
from bs4 import BeautifulSoup
import numpy
import graphlib

if __name__ == '__main__':
    print()
    # path_data = 'clean__ukraine_war_omicron_world_crisis.txt'
    path_data = 'clean__reddit_data_ukraine_test.txt'

    # filename = os.path.join(path_data, 'corpus.txt')

    keywords = ["coronavirus", "covid", "omicorn", "ukraine", "russia", "poland",
                "war", "weapons", "support", "refugees", "usa", "Volodymyr Zelenskyy",
                "Vladimir Putin", "Joe Biden", "Joe Byron", "China", "Xi Jinping",
                "Andrzej Duda", "EU", "NATO", "Oil", "Gas", "Sanction", "Subvariant"]


    with open('data/' + path_data , 'r', encoding = 'utf8') as f:
        x = f.readlines()
    texts = [h.lower().split() for h in x]


    titles = []
    for tweet in texts:
        prep = []
        for w in keywords:
            if w.lower() in tweet:
                prep.append(w)
        if prep == []:
            titles.append('ambiguous')
        else:
            # prep.join('_')
            titles.append('_'.join(prep))

    with open('titles__' + path_data, 'a') as f: # You can also print your tweets here
        for t in titles:
            f.write(t + "\n")






