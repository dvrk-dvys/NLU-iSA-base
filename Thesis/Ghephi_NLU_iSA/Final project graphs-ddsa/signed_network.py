import matplotlib.pyplot as plt
import random
import networkx as nx
from datetime import datetime


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


def unstablecount(all_signs):
    stable = 0
    unstable = 0

    for i in range(len(all_signs)):
        if (((all_signs[i]).count('+')) == 1 or ((all_signs[i]).count('+')) == 3):
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

    r = random.randint(1, 3)

    if (all_signs[ran].count('+') == 2):

        if (r == 1):
            if (g[tris_list[ran][0]][tris_list[ran][1]]['sign'] == '+'):
                g[tris_list[ran][0]][tris_list[ran][1]]['sign'] = '-'
            else:
                g[tris_list[ran][0]][tris_list[ran][1]]['sign'] = '+'

        elif (r == 2):
            if (g[tris_list[ran][1]][tris_list[ran][2]]['sign'] == '+'):
                g[tris_list[ran][1]][tris_list[ran][2]]['sign'] = '-'
            else:
                g[tris_list[ran][1]][tris_list[ran][2]]['sign'] = '+'

        else:
            if (g[tris_list[ran][0]][tris_list[ran][2]]['sign'] == '+'):
                g[tris_list[ran][0]][tris_list[ran][2]]['sign'] = '-'
            else:
                g[tris_list[ran][0]][tris_list[ran][2]]['sign'] = '+'

    else:

        if (r == 1):
            g[tris_list[ran][0]][tris_list[ran][1]]['sign'] = '+'

        elif (r == 2):
            g[tris_list[ran][1]][tris_list[ran][2]]['sign'] = '+'

        else:
            g[tris_list[ran][0]][tris_list[ran][2]]['sign'] = '+'

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
    # Create Graph from .csv ___________________________________

    analysis_start_date = '2021-01-01'
    # file_name = "Meghan,_Duchess_of_Sussex"
    # file_name = "Donald_Trump"
    # file_name = "Lil_Nas_X"
    file_name = "Joe_Biden"


    old_G = nx.Graph()
    # read_file = open("/Users/jordanharris/Downloads/wikiconflict_test.csv", "r")
    read_file = open(
        "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/" + file_name + ".csv",
        "r")

    Lines = read_file.readlines()
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

        if each[6] < 4:
            # edit : Grammar change
            counter += 1
            continue

        # edit done outside of time range
        if datetime.strptime(each[3], '%Y-%m-%d').date() <= datetime.strptime(analysis_start_date, '%Y-%m-%d').date():
            counter += 1
            continue

        if each[5] == 'ADDED':
            edge_timestamps.append((each[8], each[1], .5, each[3]))
            old_G.add_edge(each[8], each[1], weight=.5, sign='+')

        elif each[5] == 'RESTORED':
            if each[6] > 8:
                if each[8] != each[10]:
                    edge_timestamps.append((each[8], each[10], 4, each[3]))
                    old_G.add_edge(each[8], each[10], weight=4, sign='-')
                if each[8] != each[11] and each[11] != '""':
                    edge_timestamps.append((each[8], each[11], 1, each[3]))
                    old_G.add_edge(each[8], each[11], weight=1, sign='+')

        elif each[5] == 'DELETED':
            if each[6] > 8:
                if each[8] != each[10]:
                    edge_timestamps.append((each[8], each[10], 8, each[3]))
                    old_G.add_edge(each[8], each[10], weight=8, sign='-')
                if each[8] != each[11] and each[11] != '""':
                    edge_timestamps.append((each[8], each[11], 2, each[3]))
                    old_G.add_edge(each[8], each[11], weight=2, sign='+')

        counter += 1

    # 4.1.Get list of all the triangles in network________________________________________________________
    print('Original')
    nodes = old_G.nodes()
    print('Num of nodes:', len(nodes))

    # cycles = []
    cycles = nx.cycle_basis(old_G)
    print('Num of cycles:', len(cycles))

    # Remove all nodes not in a cycle_____________________________________________________________________
    simp_cyc = []

    G = nx.Graph()
    for each in cycles:
        if len(each) == 3:
            simp_cyc.append(each)

            cyc = list(range(0, len(each)))
            cyc.append(cyc[0])
            cyc_iter = []
            for iter in cyc:
                if (iter + 1) == (len(cyc) - 1):
                    cyc_iter.append((cyc[iter + 1], cyc[iter]))
                    break
                else:
                    cyc_iter.append((cyc[iter], cyc[iter + 1]))

            for z in cyc_iter:
                if (each[z[0]], each[z[1]]) not in G.edges():
                    G.add_edge(each[z[0]], each[z[1]], weight=old_G[each[z[0]]][each[z[1]]]['weight'], sign=old_G[each[z[0]]][each[z[1]]]['sign'])


    nodes = G.nodes()
    print('Num of nodes:', len(nodes))

    # cycles = []
    cycles = nx.cycle_basis(G)
    print('Num of cycles:', len(cycles))



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
    # cycles = nx.cycle_basis(G)
    # print('Num of cycles:', len(simp_cyc))

    cycles = nx.cycle_basis(old_G)
    simp_cyc = []
    for each in cycles:
        if len(each) == 3:
            simp_cyc.append(each)


    k_cores = nx.k_core(G)
    core_num_dict = nx.core_number(G)
    core_num_dict = dict(sorted(core_num_dict.items(), key=lambda item: item[1], reverse=True))
    max_k_core = max(core_num_dict.values())


    # # del_cores = 11
    # del_cores = max_k_core - 1
    # print('__________________Remove Lower K-Cores:', del_cores)
    # print('Max K Core:', max_k_core)
    #
    # # Remove nodes in a low k-core_____________________________________________________________________
    # for each in list(core_num_dict):
    #     if core_num_dict[each] <= del_cores:
    #         x = False
    #         for every in simp_cyc:
    #             if each in every:
    #                 x = True
    #         if x == True:
    #             G.remove_node(each)

    print('__________________Final Network___________________')
    # Final Nodes:
    nodes = G.nodes()
    print('Num of nodes:', len(nodes))

    # Final Cycles:
    cycles = nx.cycle_basis(G)
    # print('Num of cycles:', len(cycles))
    simp_cyc = []
    for each in cycles:
        if len(each) == 3:
            simp_cyc.append(each)
    print('Num of cycles:', len(simp_cyc))


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
    plt.savefig(file_name + '_signed_network_unbalanced.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


    # !!!!!!!!!!!!!!!!!!!Refactor!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 4.2 Store the sign details of all the triangles_____________________________________________________
    all_signs = get_signs_of_graph(G, simp_cyc)

    # 4.3.Count total number of unstable triangle in the network
    unstable = unstablecount(all_signs)

    # 5 chose the triangle in the graph that is unstable and make the triangle stable
    unstable_track = [unstable]
    # stop = int(len(all_signs)/7)
    while (unstable != 0):
        print(unstable)
        # if int(unstable) < stop:
        #     break
        g = move_graph_to_stable(G, simp_cyc, all_signs)
        all_signs = get_signs_of_graph(g, simp_cyc)
        unstable = unstablecount(all_signs)
        unstable_track.append(unstable)

    first, second = Coalition(g)
    print(len(first), first)
    print(len(second), second)

    edge_labels = nx.get_edge_attributes(g, 'sign')
    pos = nx.circular_layout(g)

    nx.draw_networkx_nodes(g, pos, nodelist=first,
                           node_color='red', node_size=100)
    nx.draw_networkx_nodes(g, pos, nodelist=second,
                           node_color='blue', node_size=100)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")

    plt.title('Balanced Signed Network: ' + file_name)
    plt.savefig(file_name + '_signed_network_balanced.png')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


    nx.write_gexf(G, "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Data/gephi/"+ file_name +"_balanced.gexf")

    coalition_A = nx.Graph()
    coalition_B = nx.Graph()

    for edge in G.edges():
        if (edge[0] in first) and (edge[1] in first):
            coalition_A.add_edge(edge[0], edge[1], sign=edge_labels[(edge[0], edge[1])])
        elif (edge[0] in second) and (edge[1] in second):
            coalition_B.add_edge(edge[0], edge[1], sign=edge_labels[(edge[0], edge[1])])
        else:
            continue


    nx.write_gexf(coalition_A, "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Final project graphs-ddsa/DDSA_Final/"+ file_name +"_coalition_A.gexf")
    nx.write_gexf(coalition_B, "/Users/jordanharris/PycharmProjects/pythonProject/upf/Data Driven Social Analytics/Final project graphs-ddsa/DDSA_Final/"+ file_name +"_coalition_B.gexf")



