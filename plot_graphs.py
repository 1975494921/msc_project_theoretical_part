import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import numpy as np


def clean_name(name):
    name = name.replace('DirectAnswer', 'TruthfulAgent')
    name = name.replace('AdversarialAnswer', 'AdversarialAgent')

    return name


def parse_graph_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = []
    pattern = re.compile(
        r'\d+: src=([A-Za-z]+Answer\([\w\d]+\)), dst=([A-Za-z]+Decision\([\w\d]+\)|[A-Za-z]+Answer\([\w\d]+\)), prob=([\d.]+)')

    for line in lines:
        match = pattern.match(line)
        if match:
            src, dst, prob = match.groups()
            parsed_data.append({'source': src, 'destination': dst, 'probability': float(prob)})
        else:
            print(f"Line didn't match: {line}")

    return pd.DataFrame(parsed_data)


def create_graph(df):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        print(row)
        src = clean_name(row['source'])
        dst = clean_name(row['destination'])
        prob = row['probability']

        if 'FinalDecision' in dst:
            if prob >= 0.5:
                G.add_edge(src, dst, weight=prob, style='solid', color='darkgreen')
            else:
                G.add_edge(src, dst, weight=prob, style='dotted', color='red')

        else:
            G.add_edge(src, dst, weight=prob, style='solid', color='blue')

    return G


def plot_graph(G):
    pos = nx.spring_layout(G, k=0.75, scale=0.9, center=(0.5, 0.5), iterations=50)

    final_decisions = [node for node in G.nodes if 'Decision' in node]
    # place the final decision nodes on the center of the plot
    for final_decision in final_decisions:
        pos[final_decision] = [0.5, 0.5]

    edges = G.edges(data=True)

    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    for edge in edges:
        src, dst, data = edge
        if 'FinalDecision' in dst:
            style = data['style']
            color = data['color']
            weight = data['weight']

            if style == 'solid':
                nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)], edge_color=color, width=2)
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)], edge_color=color, style='dotted', width=2)

    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos)

    plt.title('Graph of Connections to FinalDecision')

    plt.subplot(1, 2, 2)
    adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()
    plt.imshow(adj_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title('Adjacency Matrix of the Graph')
    plt.xticks(ticks=np.arange(len(G.nodes())), labels=G.nodes(), rotation=90)
    plt.yticks(ticks=np.arange(len(G.nodes())), labels=G.nodes())

    plt.tight_layout()
    plt.show()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot graph of connections to FinalDecision')
    parser.add_argument('--file_path', type=str, help='Path to the input file')
    args = parser.parse_args()

    # Parse the data
    df = parse_graph_data(args.file_path)

    # Create the graph
    G = create_graph(df)

    # Plot the graph
    plot_graph(G)
