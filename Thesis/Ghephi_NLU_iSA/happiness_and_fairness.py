from datetime import datetime
import networkx as nx
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from operator import itemgetter

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


def initialize_scores(G, edge_timestamps):
    fairness = {}
    goodness = {}

    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight') * 1.0 / G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness


def compute_fairness_goodness(G, edge_timestamps):
    fairness, goodness = initialize_scores(G, edge_timestamps)

    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0

        print('-----------------')
        print ("Iteration number", iter)

        print ('Updating goodness')
        for node in nodes:

            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]] * edge[2]

            try:
                dg += abs(g / len(inedges) - goodness[node])
                goodness[node] = g / len(inedges)
            except:
                pass

        print('Updating fairness')
        for node in nodes:
            out_edges = G.out_edges(node, data='weight')
            f = 0
            for edge in out_edges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]]) / 2.0
            try:
                df += abs(f / len(out_edges) - fairness[node])
                fairness[node] = f / len(out_edges)
            except:
                pass

        print('Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter += 1

    return fairness, goodness

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Create Graph from .csv ___________________________________

    analysis_start_date = '2021-01-01'
    file_name = "Joe_Biden"
    # file_name = "Meghan,_Duchess_of_Sussex"
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
                    # Your deletion is redone after it was undone
                    edge_timestamps.append((each[8], each[11], 1.5, each[3]))
                    G.add_edge(each[8], each[11], weight=1.5)

        counter += 1

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, edge_timestamps)
    fairness = dict(sorted(fairness.items(), key=lambda item: item[1], reverse=True))
    goodness = dict(sorted(goodness.items(), key=lambda item: item[1], reverse=True))


    avg_node_weights = {}
    for each in G.nodes:
        w = 0
        out = G.out_edges(each)
        for edge in out:
            w += G.get_edge_data(edge[0], edge[1], default=None)['weight']
        if len(out) == 0:
            continue
        avg_node_weights[each] = w/len(out)

        # node_lables.append(each)
        # x_axis_weights.append(w/len(out))
    avg_node_weights = dict(sorted(avg_node_weights.items(), key=lambda item: item[1], reverse=True))

    test = list(avg_node_weights.items())
    fin_avg_weights = []
    for each in test:
        fin_avg_weights.append(each[1])

    node_lables = list(avg_node_weights.keys())

    print('Fairness', fairness)
    print('goodness', goodness)
    print('-----------------')

    y_fairness = []
    for each in node_lables:
        y_fairness.append(fairness[each])

    y_goodness = []
    for each in node_lables:
        y_goodness.append(goodness[each])




    # _________________________________________________

    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(fin_avg_weights, y_fairness, label='True Position')
    plt.axvline(x=0)
    # Title & Subtitle
    plt.title('User adjacency Fairness Score vs Avg. Out-Edge Weight')
    plt.ylabel("Fairness")
    plt.xlabel("Avg. Weight")
    filename = 'scatter'
    plt.savefig(r'/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Final project graphs-ddsa/DDSA_Final/fairness_weight_' + filename + '.png')
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(fin_avg_weights, y_goodness, label='True Position')
    plt.axvline(x=0)
    # Title & Subtitle
    plt.title('User adjacency Goodness vs Avg. Out-Edge Weight')
    plt.ylabel("Goodness")
    plt.xlabel("Avg. Weight")
    filename = 'scatter'
    plt.savefig(
        r'/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Final project graphs-ddsa/DDSA_Final/goodness_weight_' + filename + '.png')
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()


    # _________________________________________________


    # z_goodness = sorted(y_goodness, reverse=True)
    #
    #
    # z = list(range(0, len(node_lables)))
    # # z = sorted(edge, key=itemgetter(2))
    #
    # data = pd.DataFrame(data={'x': node_lables, 'y': fin_avg_weights, 'z': z_goodness})
    # data = data.pivot(index='x', columns='y', values='z')
    # sns.heatmap(data, cbar_kws = dict(use_gridspec=False,location="top"))
    # plt.show()
    #

  # _________________________________________________
    y_coords = []
    x_coords = []
    happy_nodes = {}

    for n in G.nodes:
        y_coords.append(fairness[n])
        x_coords.append(goodness[n])
        happy_nodes[n] = (goodness[n], fairness[n])


    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    coef = np.polyfit(x_coords, y_coords, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(x_coords, y_coords, 'yo', x_coords, poly1d_fn(x_coords), '--k')

    # Title & Subtitle
    plt.title('User adjacency Goodness vs User adjacency Goodness')
    plt.ylabel("Fairnness")
    plt.xlabel("Goodness")
    filename = 'scatter'
    plt.savefig(
        r'/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Final project graphs-ddsa/DDSA_Final/goodness_fairness_' + filename + '.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()



    # pos_fairness = {}
    # for key in fairness:
    #     if (fairness[key] >= 0):
    #         pos_fairness[key] = fairness[key]
    #
    # pos_goodness = {}
    # for key in goodness:
    #     if (goodness[key] >= 0):
    #         pos_goodness[key] = goodness[key]


    active_nodes = dict(sorted(happy_nodes.items(), key=lambda item: item[1], reverse=True))
    print('Best Nodes:', file_name)

    count = 0
    for key, value in active_nodes.items():
        count+= 1
        print(key, ' : ', value)
        if count == 10:
            break

    print('-----------------')


    # print('Fairness', pos_fairness)
    # print('goodness', pos_goodness)



#
#  much this node is liked/trusted by other nodes,
# while the fairness of a node captures how fair the node is in rating other nodes
# ' likeability or trust level.

