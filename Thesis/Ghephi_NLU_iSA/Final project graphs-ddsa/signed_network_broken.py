import pandas as pd
# import re
import csv
import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
# from pyparsing import *
# import string, re


# remember the author info hasent been assessed

def get_signs_of_graph(g, tris_list):
    # eg-['A-B','B-C','C-A']
    all_signs = []
    trip_nodes = []
    trip_sets = []

    for i in range(len(tris_list)):
        t = []

        cyc = list(range(0, len(tris_list[i])))
        cyc.append(cyc[0])
        cyc_iter = []
        for iter in cyc:
            cyc_iter.append((cyc[iter], cyc[iter + 1]))
            if (iter + 1) == (len(cyc) - 1):
                break

        # len_cyc = list(itertools.combinations(range(0, len(tris_list[i])), 2))
        for index in cyc_iter:
             if tris_list[i][index[0]] in g[tris_list[i][index[1]]]:
                 t.append(g[tris_list[i][index[0]]][tris_list[i][index[1]]]['sign'])

        all_signs.append(t)
        trip_nodes = trip_nodes + tris_list[i]
        trip_sets.append(tris_list[i])
    return all_signs, set(trip_nodes), trip_sets


def unstablecount(all_signs):
    stable = 0
    unstable = 0

    for i in range(len(all_signs)):
        max_elem = max(set(all_signs[i]), key=all_signs[i].count)
        if (all_signs[i].count('+') == (len(all_signs[i]))) or (max_elem == '-'):
           stable += 1

    unstable = len(all_signs) - stable
    return unstable


def move_graph_to_stable(g, tris_list, all_signs):
    found_unstable = False
    ran = 0

    while (found_unstable == False):
        ran = random.randint(0, len(tris_list) - 1)
        if (all_signs[ran].count('+') % 2 == 0):
            found_unstable = True
        else:
            continue

    r = random.randint(0, len(all_signs[ran]) - 1)
    cyc = list(range(0, len(tris_list[ran])))
    cyc.append(cyc[0])
    cyc_iter = []
    for iter in cyc:
        if (iter + 1) == (len(cyc) - 1):
            cyc_iter.append((cyc[iter + 1], cyc[iter]))
            break
        else:
            cyc_iter.append((cyc[iter], cyc[iter + 1]))

    max_elem = max(set(all_signs[ran]), key=all_signs[ran].count)

    #if (max_elem == '+') and (all_signs[ran].count('+') < len(all_signs[ran])):
    if (max_elem == '+') and (all_signs[ran].count('+') == len(all_signs[ran]) - 1):

        if (g[tris_list[ran][cyc_iter[r][0]]][tris_list[ran][cyc_iter[r][1]]]['sign'] == '+'):
            g[tris_list[ran][cyc_iter[r][0]]][tris_list[ran][cyc_iter[r][1]]]['sign'] = '-'
        else:
            g[tris_list[ran][cyc_iter[r][0]]][tris_list[ran][cyc_iter[r][1]]]['sign'] = '+'

    else:
        g[tris_list[ran][cyc_iter[r][0]]][tris_list[ran][cyc_iter[r][1]]]['sign'] = '+'

    return g




def Coalition(g):
    f = []
    s = []
    nodes = g.nodes()
    r = random.choice(list(nodes))

    f.append(r)
    processed_nodes = []
    to_be_processed = [r]

    for each in to_be_processed:
        if each not in processed_nodes:
            neigh = list(g.neighbors(each))
            for i in range(len(neigh)):
                if (g[each][neigh[i]]['sign'] == '+'):
                    if (neigh[i] not in f):
                        f.append(neigh[i])
                    if (neigh[i] not in to_be_processed):
                        to_be_processed.append(neigh[i])
                elif (g[each][neigh[i]]['sign'] == '-'):
                    if (neigh[i] not in s):
                        s.append(neigh[i])
                        processed_nodes.append(neigh[i])

            processed_nodes.append(each)

    return f, s

if __name__ == '__main__':

    G = nx.Graph()
    # read_file = open("/Users/jordanharris/Downloads/wikiconflict_test.csv", "r")
    read_file = open(
        "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/Meghan,_Duchess_of_Sussex.csv",
        "r")
    # read_file = open(
    #     "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/Donald_Trump.csv", "r")

    Lines = read_file.readlines()
    read_file.close()


    read_file = open(
        "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/triplet_nodes.txt", "r")

    triplet_nodes = read_file.readlines()
    pre = []
    for each in triplet_nodes:
        pre.append(each[1:][:-2])
    read_file.close()


    index = 0
    # Strips the newline character
    for line in Lines:
        x = line.replace('\n', '').split(';')
        Lines[index] = x
        break

    # Add edges weight and sign ___________________________________

    for line in Lines:
        if index == 0:
            index += 1
            continue

        x = line.replace('\n', '').split(';')

        # Overall Edit Wars
        spot = 0

        # Individual Pages
        for each in x:
            if spot in [0, 3, 5, 8, 9, 10, 11]:
                x[spot] = str(each)
                spot += 1
            else:
                x[spot] = int(each)
                spot += 1

        Lines[index] = x
        index += 1

    counter = 0
    for each in Lines:
        # column title
        if counter == 0:
            counter += 1
            # print(counter)
            continue

        c = 0
        if len(each) != 12:
            c += 1
            counter += 1
            # print(c)
            continue

        # # 95373
        # # 66000
        # if counter == 66000:
        #     break

        # 248
        # if len(G.nodes) == 20:
        #     break
        #
        # if (each[8] not in triplet_nodes) and (each[10] not in triplet_nodes) and (each[11] not in triplet_nodes):
        #     counter += 1
        #     continue


        if each[6] < 4:
            # edit : Grammar change
            counter += 1
            continue

        if each[5] == 'ADDED':
            G.add_edge(each[8], each[1], weight=.5, sign='+')
            counter += 1
        if each[5] == 'RESTORED':
            if each[6] > 20:
                if each[8] != each[10]:
                    G.add_edge(each[8], each[10], weight=-1.5, sign='-')
                if each[8] != each[11]:
                    G.add_edge(each[8], each[11], weight=2, sign='+')
        #     else:
        #         if each[8] != each[10]:
        #             G.add_edge(each[8], each[10], weight=-1.5, sign='-')
        #         if each[8] != each[11]:
        #             G.add_edge(each[8], each[11], weight=1, sign='+')
        # elif each[5] == 'DELETED':
            if each[6] > 20:
                if each[8] != each[10]:
                    G.add_edge(each[8], each[10], weight=-2, sign='-')
                if each[8] != each[11]:
                    G.add_edge(each[8], each[11], weight=1.5, sign='+')
            # else:
            #     if each[8] != each[10]:
            #         G.add_edge(each[8], each[10], weight=-1.5, sign='-')
            #     if each[8] != each[11]:
            #         G.add_edge(each[8], each[11], weight=1, sign='+')

        counter += 1

    # 4.1.Get list of all the triangles in network________________________________________________________
    print('Original')
    nodes = G.nodes()
    print('Num of nodes:', len(nodes))

    # cycles = []
    cycles = nx.cycle_basis(G)
    print('Num of cycles:', len(cycles))


    # Remove all nodes not in a triplet_____________________________________________________________________
    for each in list(nodes):
        x = False
        for every in cycles:
            if each in every:
                x = True
        if x == False:
            G.remove_node(each)

    print('_________________Trim Nodes not in a Cycle____________________')
    nodes = G.nodes()
    print('Num of nodes:', len(nodes))
    cycles = nx.cycle_basis(G)
    print('Num of cycles:', len(cycles))

    # y = sorted(nx.connected_components(G), key=len, reverse=True)
    # largest = max(nx.connected_components(G), key=len)
    #
    # for node in list(nodes):
    #     if node not in largest:
    #         G.remove_node(node)
    # nodes = G.nodes()
    # print('Num of nodes:', len(nodes))


    k_cores = nx.k_core(G)
    core_num_dict = nx.core_number(G)
    core_num_dict = dict(sorted(core_num_dict.items(), key=lambda item: item[1], reverse=True))
    max_k_core = max(core_num_dict.values())


    del_cores = 10
    print('__________________Remove Lower K-Cores:', del_cores)
    print('Max K Core:', max_k_core)

    # Remove nodes in a low k-core_____________________________________________________________________
    for each in list(core_num_dict):
        if core_num_dict[each] <= del_cores:
            x = False
            for every in cycles:
                if each in every:
                    x = True
            if x == True:
                G.remove_node(each)

    print('__________________Final Network___________________')
    # Final Nodes:
    nodes = G.nodes()
    print('Num of nodes:', len(nodes))

    # Final Cycles:
    cycles = nx.cycle_basis(G)
    print('Num of cycles:', len(cycles))

    # Final Cores:
    k_cores = nx.k_core(G)
    core_num_dict = nx.core_number(G)
    core_num_dict = dict(sorted(core_num_dict.items(), key=lambda item: item[1], reverse=True))

    # Display graph______________________________________________________________________________________
    edge_attributes = nx.get_edge_attributes(G, 'sign')
    pos = nx.circular_layout(G)
    nx.draw(G, pos, node_size=12, with_labels=1)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_attributes, font_size=8, font_color='blue')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.savefig('signed_network_unbalanced.png')
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()


    # !!!!!!!!!!!!!!!!!!!Refactor!!!!!!!!!!!!!!!!!!!!!!!!!!

    # tris_list = [list(x) for x in itertools.combinations(nodes, 3)]

    # 4.2 Store the sign details of all the triangles_____________________________________________________
    all_signs, triplets, trip_sets = get_signs_of_graph(G, cycles)

    # 4.3.Count total number of unstable triangle in the network
    unstable = unstablecount(all_signs)

    # 5 chose the triangle in the graph that is unstable and make the triangle stable
    unstable_track = [unstable]
    stop = int(len(all_signs)/3)
    while (unstable != 0):
        print(unstable)
        if int(unstable) < stop:
            break
        g = move_graph_to_stable(G, cycles, all_signs)
        all_signs, triplets, trip_sets = get_signs_of_graph(g, cycles)
        unstable = unstablecount(all_signs)
        unstable_track.append(unstable)

    first, second = Coalition(g)
    print(first)
    print(second)

    edge_labels = nx.get_edge_attributes(g, 'sign')
    pos = nx.circular_layout(g)

    nx.draw_networkx_nodes(g, pos, nodelist=first,
                           node_color='red', node_size=100)
    nx.draw_networkx_nodes(g, pos, nodelist=second,
                           node_color='blue', node_size=100)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")
    plt.savefig('signed_network_balanced.png')

    # plt.savefig(r'C:/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/test.png')
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    # y = sorted(nx.connected_components(G), key=len, reverse=True)
    # largest = max(nx.connected_components(G), key=len)
    #
    # for node in list(G.nodes):
    #     if node not in largest:
    #         G.remove_node(node)

    f = open("triplet_nodes.txt", "w+")

    f = open(
        "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/triplet_nodes.txt", "a")


    for test in triplets:
        f.write(str(test) + '\n')

    # print('test')
    # nx.write_gexf(G, "Donald_Trump_graph.gexf")

    # read_file = open("/Users/jordanharris/Downloads/wikiconflict_test.txt", "w")
    # read_file.writelines(Lines)
    # read_file.close()


