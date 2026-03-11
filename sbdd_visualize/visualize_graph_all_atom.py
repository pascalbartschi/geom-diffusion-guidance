import argparse
from torch_geometric.utils import remove_self_loops
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.io as io
import torch
import re
import numpy as np
from moldiff.constants import amino_acid_colors, atom_colors, atom_decoder, aa_decoder3



def visualize_graph(graph, 
                    title,
                    highlight=None,
                    highlight_symbol = 'x',
                    show_edges = True,
                    show_edge_attr=False, 
                    add_traces = [], 
                    linewidth=2, 
                    markersize=3, 
                    remove_mn_edges=False,
                    remove_noncov_edges = False, 
                    atom_level_protein_structure = False):

    # Remove the master node edges
    if remove_mn_edges:
        master_node_index = torch.max(graph.edge_index)
        #mask = graph.edge_index[1] != master_node_index
        mask = torch.all(graph.edge_index != master_node_index, dim=0)
        graph.edge_index = graph.edge_index[:, mask]

    # Remove the edges between atoms and AAs
    if remove_noncov_edges:
        mask1 = (graph.edge_index > graph.n_nodes[1]-1).any(dim=0)
        mask2 = (graph.edge_index == graph.n_nodes[0]).any(dim=0)
        mask = ~mask1 | mask2
        graph.edge_index = graph.edge_index[:, mask]

        print_columns_as_tuple(graph.edge_index)

        # mask1 = graph.edge_index[0] < graph.n_nodes[1].item()
        # mask2 = graph.edge_index[1] < graph.n_nodes[1].item()
        # combined_mask = mask1 & mask2


    if show_edge_attr:
        edge_list, edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        edge_list = edge_list.T.tolist()
        edge_attr = edge_attr.tolist()

        edges=[]
        hoverinfo_edges = []
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))
                hoverinfo_edges.append(edge_attr[idx])

        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_nodes = graph.x[:,9:40].tolist()

        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]

        for n in range (len(hoverinfo_edges)):
            hoverinfo_edges[n] = [int(entry) if entry % 1 == 0 else round(entry, 4) for entry in hoverinfo_edges[n]]


    else: 
        edge_list, _ = remove_self_loops(graph.edge_index)
        edge_list = edge_list.T.tolist()
    
        edges=[]
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))


        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_edges = 'None'
        hoverinfo_nodes = graph.x[:,9:40].tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]

    N = graph.x.shape[0]
    print(N)


    #MARKER COLORS AND LABELS
    #------------------------------------------------------------------------------------------------------------------------------------

    hoverinfo_nodes = [0 for i in range(N)]
    markercolor = [0 for i in range(N)]
    markersize = [markersize for i in range(N)]


    index, atomtypes = (graph.x[:,:10] == 1).nonzero(as_tuple=True) #identify the index of the first 1 in the feature matrix = atom type
    print(atomtypes.tolist())
    for idx, atomtype in zip(index.tolist(), atomtypes.tolist()):

        hoverinfo_nodes[idx] = atom_decoder[atomtype] # If the chemical element should be displayed
        #hoverinfo_nodes[idx] = int(markersize[idx]) # If the attention score should be displayed
        markercolor[idx] = atom_colors[atom_decoder[atomtype]]
        markersize[idx] = 20


    index, aa_types = (graph.x[:,10:] == 1).nonzero(as_tuple=True) # identify the index of the first 1 in the feature matrix = aa type
    print([aa for aa in aa_types.tolist()])
    aa_names = [aa_decoder3[i] for i in aa_types.tolist()]


    for idx, aa_type, aa_name in zip(index.tolist(), aa_types.tolist(), aa_names):

        hoverinfo_nodes[idx] = aa_decoder3[aa_type]# + f'({int(markersize[idx])})'
        markercolor[idx] = amino_acid_colors[aa_name]


    print(markercolor)
    print(markersize)
    #-----------------------------------------------------------------------------------------------------------------------------------


    #Marker shape based on atom type (Fe, Cl and the atoms in highligh as crosses, the rest as circles)
    not_ions = [0,1,2,3,4,5,6,7,8]
    # symbols = ['x' if atom not in not_ions else 'circle' for atom in atomtypes]
    symbols = ['circle' for atom in atomtypes]
    if highlight != None:
        symbols = [highlight_symbol if index in highlight else symbol for index, symbol in enumerate(symbols)]


    # Prepare the coordinates of the nodes and edges
    atomcoords = graph.pos.numpy()

    Xn=[atomcoords[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[atomcoords[k][1] for k in range(N)]# y-coordinates
    Zn=[atomcoords[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]

    #print(edges)

    for e in edges:
        Xe+=[atomcoords[e[0]][0],atomcoords[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[atomcoords[e[0]][1],atomcoords[e[1]][1], None]# y-coordinates of edge ends
        Ze+=[atomcoords[e[0]][2],atomcoords[e[1]][2], None]# z-coordinates of edge ends


    # Configure Plot, trace1 = edges, trace2 = nodes
    trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(50,50,50)', width=0.2),
               text=hoverinfo_edges,
               #textposition= 'middle center',
               hoverinfo = 'text'
               )

    trace2=go.Scatter3d(x=Xn,
                    y=Yn,
                    z=Zn,
                    mode='markers',
                    marker=dict(symbol=symbols,
                                size=markersize,
                                color=markercolor,
                                #colorscale = 'viridis',
                                line=dict(color='rgb(50,50,50)', width=0.2)
                                ),
                    text=hoverinfo_nodes,
                    hoverinfo='text'
                    )
    
    trace3=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='text',
                textfont=dict(size=16),
                text=hoverinfo_nodes,
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title=title,
            width=2000,
            height=2000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
    )

    # Add the traces that are given in add_traces
    if show_edges:
        data = [trace1, trace2, trace3] + add_traces
    else: 
        data = [trace2, trace3] + add_traces


    # PLOT
    fig=go.Figure(data=data, layout=layout)
    io.renderers.default='notebook'  # Set the default renderer to browser
    io.show(fig)  # This will open in a new browser tab




def visualize_graph_all_atom(
        graph,
        title,
        highlight=None,
        highlight_symbol='x',
        show_edges=True,
        show_edge_attr=False,
        add_traces=[],
        linewidth=2,
        markersize=3,
        remove_mn_edges=False,
        remove_noncov_edges=False,
        full_atom_level=True):
    """
    Visualize ligand–protein graphs with option for CA-only or full-atom protein representation.
    Ligand nodes are always atom-level. Protein nodes carry both atom and residue labels.

    Args:
        full_atom_level (bool): If True, show all protein atoms. If False, only CA atoms + ligand atoms.
    """

    # ----------- Node selection -----------
    if not full_atom_level:
        mask = graph.lig_mask | (graph.prot_mask & graph.ca_mask)
    else:
        mask = torch.ones(graph.x.size(0), dtype=torch.bool, device=graph.x.device)

    # mapping old→new node indices
    old_to_new = -torch.ones(mask.size(0), dtype=torch.long, device=graph.x.device)
    old_to_new[mask] = torch.arange(mask.sum(), device=graph.x.device)

    # masked node features and positions
    x = graph.x[mask]
    pos = graph.pos[mask]
    N = x.shape[0]

    # ----------- Edge filtering & remapping -----------
    edge_index = graph.edge_index
    # keep only edges where both nodes survive
    valid_edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    edge_index = edge_index[:, valid_edge_mask]
    # reindex to new compact node indices
    edge_index = old_to_new[edge_index]

    # optional edge removals
    if remove_mn_edges:
        master_node_index = torch.max(edge_index)
        edge_mask = torch.all(edge_index != master_node_index, dim=0)
        edge_index = edge_index[:, edge_mask]

    if remove_noncov_edges:
        mask1 = (edge_index > graph.n_nodes[1]-1).any(dim=0)
        mask2 = (edge_index == graph.n_nodes[0]).any(dim=0)
        edge_mask = ~mask1 | mask2
        edge_index = edge_index[:, edge_mask]

    # remove self-loops
    edge_list, _ = remove_self_loops(edge_index)
    edge_list = edge_list.T.tolist()
    edges = []
    for pair in edge_list:
        if (pair[1], pair[0]) not in edges:
            edges.append((pair[0], pair[1]))

    # ----------- Node decoding (atom + residue levels) -----------
    hoverinfo_nodes = ["" for _ in range(N)]
    markercolor = [0 for _ in range(N)]
    markersize_list = [markersize for _ in range(N)]

    # ligand atom types
    idx, atomtypes = (x[:, :len(atom_decoder)] == 1).nonzero(as_tuple=True)
    for i, t in zip(idx.tolist(), atomtypes.tolist()):
        atom_name = atom_decoder[t]
        hoverinfo_nodes[i] = atom_name
        markercolor[i] = atom_colors.get(atom_name, "grey")
        markersize_list[i] = 20

    # protein atom types
    start = len(atom_decoder)
    idx, patomtypes = (x[:, start:start+len(atom_decoder)] == 1).nonzero(as_tuple=True)
    for i, t in zip(idx.tolist(), patomtypes.tolist()):
        atom_name = atom_decoder[t]
        hoverinfo_nodes[i] = atom_name
        markercolor[i] = atom_colors.get(atom_name, "grey")

    # protein residue types
    start2 = start + len(atom_decoder)
    idx, aatypes = (x[:, start2:start2+len(aa_decoder3)] == 1).nonzero(as_tuple=True)
    for i, t in zip(idx.tolist(), aatypes.tolist()):
        aa_name = aa_decoder3[t]
        if hoverinfo_nodes[i] != "":
            hoverinfo_nodes_i = f"{hoverinfo_nodes[i]}" if full_atom_level else f"{hoverinfo_nodes[i]} ({aa_name})"
            hoverinfo_nodes[i] = hoverinfo_nodes_i
        else:
            hoverinfo_nodes[i] = f"CA ({aa_name})"
        markercolor[i] = "grey" # amino_acid_colors.get(aa_name, "lightgrey")

    # ----------- Node/edge coordinates -----------
    atomcoords = pos.cpu().numpy()
    Xn = atomcoords[:, 0].tolist()
    Yn = atomcoords[:, 1].tolist()
    Zn = atomcoords[:, 2].tolist()

    Xe, Ye, Ze = [], [], []
    for e in edges:
        Xe += [atomcoords[e[0]][0], atomcoords[e[1]][0], None]
        Ye += [atomcoords[e[0]][1], atomcoords[e[1]][1], None]
        Ze += [atomcoords[e[0]][2], atomcoords[e[1]][2], None]

    # ----------- Traces -----------
    trace_edges = go.Scatter3d(
        x=Xe, y=Ye, z=Ze,
        mode='lines',
        line=dict(color='rgb(50,50,50)', width=0.5),
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=Xn, y=Yn, z=Zn,
        mode='markers',
        marker=dict(size=markersize_list,
                    color=markercolor,
                    line=dict(color='black', width=0.2)),
        text=hoverinfo_nodes,
        hoverinfo='text'
    )

    trace_labels = go.Scatter3d(
        x=Xn, y=Yn, z=Zn,
        mode='text',
        text=hoverinfo_nodes,
        textfont=dict(size=12, color="black"),
        hoverinfo='none'
    )

    layout = go.Layout(
        title=title,
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(t=100),
        hovermode='closest'
    )

    # Assemble final traces
    if show_edges:
        data = [trace_edges, trace_nodes, trace_labels] + add_traces
    else:
        data = [trace_nodes, trace_labels] + add_traces

    fig = go.Figure(data=data, layout=layout)
    # io.show(fig)
    # fig.show(renderer="browser")
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)

    # # --- choose camera for "horizontal" look ---
    # coords = atomcoords  # shape (N,3) numpy array already
    # span = coords.max(axis=0) - coords.min(axis=0)
    # sx, sy, sz = span.tolist()

    # # Put the longest in-plane axis horizontally by viewing roughly along the other axis
    # # If structure is taller than wide (sy > sx), rotate view by 90deg around z.
    # if sy > sx:
    #     eye = dict(x=-0.2, y=1.8, z=0.8)  # your "horizontal" view (rotated)
    # else:
    #     eye = dict(x=1.8, y=-0.2, z=0.8)  # the 90deg counterpart

    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=-0.2, y=1.8, z=0.8),
                up=dict(x=1, y=0, z=0)   
            )
        )
    )

    fig.write_html(f"{safe_title}.html")
    fig.write_image(f"{safe_title}.png", width=3000, height=2500, scale=3)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, required=True)
    parser.add_argument("--full_atom", action="store_true")
    args = parser.parse_args()

    graph = torch.load(args.graph_path)
    full_atom = args.full_atom if args.full_atom else False

    visualize_graph_all_atom(graph, title='1a1e_graph', full_atom_level = full_atom)

    # e.g., 
    # python visualize_graph_all_atom.py --graph_path ../bioinformatics/data_extracted_graphs_atom_level/1a1e.pt