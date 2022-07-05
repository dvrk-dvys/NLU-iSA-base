from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# remember the author info hasent been assessed

def get_signs_of_graph(g, tris_list):
    # eg-['A-B','B-C','C-A']
    all_signs = []

    for i in range(len(tris_list)):
        t = []
        t.append(g[tris_list[i][0]][tris_list[i][1]]['sign'])
        t.append(g[tris_list[i][1]][tris_list[i][2]]['sign'])
        t.append(g[tris_list[i][2]][tris_list[i][0]]['sign'])
        all_signs.append(t)
    return all_signs

if __name__ == '__main__':

    # Create Graph from .csv ___________________________________

    analysis_start_date = '2021-01-01'
    file_name = "Joe_Biden"
    # file_name = "Meghan,_Duchess_of_Sussex"
    # file_name = "Donald_Trump"
    # file_name = "Lil_Nas_X"

    G = nx.DiGraph()
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
                    edge_timestamps.append((each[8], each[10], 4, each[3]))
                    G.add_edge(each[8], each[10], weight=4)
                if each[8] != each[11] and each[11] != '""':
                    # test += 1

                    # your content is restored
                    edge_timestamps.append((each[8], each[11], 1, each[3]))
                    G.add_edge(each[8], each[11], weight=1)

        elif each[5] == 'DELETED':
            if each[6] > 8:
                if each[8] == '"Sampajanna"' and each[10] == '"Keivan.f"':
                    print()

                if each[8] != each[10]:
                    # test += 1

                    edge_timestamps.append((each[8], each[10], 8, each[3]))
                    # Your post is deleted, worst thing
                    G.add_edge(each[8], each[10], weight=8)
                if each[8] != each[11] and each[11] != '""':
                    # test += 1

                    # Your deletion is redone after it was undone
                    edge_timestamps.append((each[8], each[11], 2, each[3]))
                    G.add_edge(each[8], each[11], weight=2)
        # if test == 0 and each[6] > 8 and (each[8] != each[11] and each[8] != each[10]):
        #     print()
        counter += 1

    nodes = []
    for each in G.nodes:
        nodes.append(each)

    print(len(nodes), len(edge_timestamps))

    # captture nodes in 2 separate lists
    node_list_1 = []
    node_list_2 = []

    for i in tqdm(edge_timestamps):
        node_list_1.append(i[0])
        node_list_2.append(i[1])

    fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    # plot graph
    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)
    plt.savefig(file_name + 'new_link_orediction.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))

    # build adjacency matrix
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)


    print(adj_G.shape)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if i != j:
                if nx.shortest_path_length(G, node_list[i], node_list[j]) <= 2:
                    if adj_G[i, j] == 0:
                        all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    len(all_unconnected_pairs)

    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0

    initial_node_count = len(G.nodes)

    fb_df_temp = fb_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(fb_df.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index=i)

    initial_node_count = len(G.nodes)

    fb_df_temp = fb_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(fb_df.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index=i)

    # create dataframe of removable edges
    fb_df_ghost = fb_df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    # create dataframe of removable edges
    fb_df_ghost = fb_df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    # drop removable edges
    fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

    # Generate walks
    node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

    # train node2vec model
    n2w_model = node2vec.fit(window=7, min_count=1)

    x = [(n2w_model[str(i)] + n2w_model[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]

    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)

    lr = LogisticRegression(class_weight="balanced")

    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    print('PREDICTIONS:', predictions)
    roc_auc_val = roc_auc_score(ytest, predictions[:, 1])
    print('ROC_AUC SCORE:', roc_auc_val)

    print('_____________________________________')

