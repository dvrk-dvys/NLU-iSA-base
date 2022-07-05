import sys

import networkx as nx
import math
import pandas as pd
import re
import csv
import networkx as nx
from datetime import datetime
import dateparser
from csv import writer

def initialize_scores(G):
    fairness = {}
    goodness = {}
    
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0

        print('-----------------')
        print("Iteration number", iter)
        
        print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]["weight"]

            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass

        print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        print('Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter+=1
    
    return fairness, goodness

skip = int(sys.argv[1])

G = nx.DiGraph()

f = open("network.csv","r")
for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) ## the weight should already be in the range of -1 to 1
f.close()


# these two dictionaries have the required scores
fairness, goodness = compute_fairness_goodness(G)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Create Graph from .csv ___________________________________

    analysis_start_date = '2021-01-01'
    file_name = "Joe_Biden"
    # file_name = "Meghan,_Duchess_of_Sussex"
    # file_name = "Donald_Trump"
    # file_name = "Lil_Nas_X"

    G = nx.Graph()
    read_file = open(
        "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/" + file_name + ".csv",
        "r")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    analysis_start_date = '2021-01-01'
    topic = 'Joe_Biden'

    Lines = read_file.readlines()

    index = 0
    # Strips the newline character
    for line in Lines:
        x = line.replace('\n', '').split(';')
        Lines[index] = x
        break

    for line in Lines:
        if index == 0:
            index += 1
            continue

        x = line.replace('\n', '').split(';')

        spot = 0

        # Individual Pages
        for each in x:
            if spot in [0, 5, 8, 9, 10, 11]:
                x[spot] = str(each)
                spot += 1
            elif spot == 3:
                # 2018-09-15
                x[spot] = each[0:10]
                spot += 1
            else:
                x[spot] = int(each)
                spot += 1

        Lines[index] = x
        index += 1

    # 'Source', 'Target', 'Weight', 'Time'    '2021-03-18T11:53:25Z'
    edge_timestamps = []
    counter = 0
    revisionIDs = []
    for each in Lines:
        c = 0
        if len(each) != 12:
            c += 1
            print('Amt of Data Errors:', c)
            continue

        # title
        if counter == 0:
            counter += 1
            continue

        # grammar change
        if each[6] < 4:
            # grammar change
            counter += 1
            continue

        # edit done outside of time range
        if datetime.strptime(each[3], '%Y-%m-%d').date() <= datetime.strptime(analysis_start_date, '%Y-%m-%d').date():
            counter += 1
            continue

        # test = 0
        if each[5] == 'ADDED':
            # test +=1
            edge_timestamps.append((each[8], each[1], .5, each[3]))
            G.add_edge(each[8], each[1], weight=.5)
            # counter += 1
        elif each[5] == 'RESTORED':
            if each[6] > 8:
                if each[8] != each[10]:
                    # test += 1

                    # Your deletion is undone
                    edge_timestamps.append((each[8], each[10], -1.5, each[3]))
                    G.add_edge(each[8], each[10], weight=-1.5)
                if each[8] != each[11] and each[11] != '""':
                    # test += 1

                    # your content is restored
                    edge_timestamps.append((each[8], each[11], 2, each[3]))
                    G.add_edge(each[8], each[11], weight=2)

        elif each[5] == 'DELETED':
            if each[6] > 8:
                if each[8] == '"Sampajanna"' and each[10] == '"Keivan.f"':
                    print()

                if each[8] != each[10]:
                    # test += 1

                    edge_timestamps.append((each[8], each[10], -2, each[3]))
                    # Your post is deleted, worst thing
                    G.add_edge(each[8], each[10], weight=-2)
                if each[8] != each[11] and each[11] != '""':
                    # test += 1

                    # Your deletion is redone after it was undone
                    edge_timestamps.append((each[8], each[11], 1.5, each[3]))
                    G.add_edge(each[8], each[11], weight=1.5)
        # if test == 0 and each[6] > 8 and (each[8] != each[11] and each[8] != each[10]):
        #     print()
        counter += 1

    print()

