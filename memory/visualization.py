import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#101923"
FG = "#b9c8d8"

def style(ax):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#293849")
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

def map_figure(session, truth=False):
    import networkx as nx
    fig, ax = plt.subplots(figsize=(6,4.5))
    style(ax)
    graph = session.env.dungeon.graph if truth else session.trace.graph
    current = session.env.position if truth else session.trace.position
    positions = {p:(p[0],-p[1]) for p in graph}
    colors = ["#60e1c1" if p == current else "#bc9565" if graph.nodes[p].get("seal",False) or (truth and p == session.env.dungeon.exit) else "#263c50" for p in graph]
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#526c83", width=2)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=420, edgecolors="#7b95ac")
    nx.draw_networkx_labels(graph, positions, labels={p:str(i+1) for i,p in enumerate(graph)}, ax=ax, font_color="#f1f6fc", font_size=8)
    # Local door stubs reveal only exits actually observed, no remote endpoint.
    if not truth:
        from dungeon.entities import DELTAS, COLORS
        palette = ["#ef7272","#71abff","#d9b574","#b89aff"]
        for p, data in graph.nodes(data=True):
            for di, status in enumerate(data.get("passages",())):
                if status >= 2 and status <= 5:
                    dx,dy = DELTAS[di]
                    ax.plot([p[0],p[0]+dx*.36],[-p[1],-p[1]-dy*.36], color=palette[status-2], linewidth=5)
    ax.set_title("GROUND TRUTH · privileged debug view" if truth else "DISCOVERED MAP · observation-derived")
    ax.margins(.2)
    ax.axis("off")
    fig.tight_layout()
    return fig

def hidden_figure(timeline):
    fig, axes = plt.subplots(1,2,figsize=(9,3), gridspec_kw={"width_ratios":[2,1]})
    for ax in axes:
        style(ax)
    vectors = [t["hidden"] for t in timeline if t["hidden"]]
    if vectors:
        a = np.asarray(vectors)
        axes[0].imshow(a.T, aspect="auto", cmap="coolwarm", vmin=-max(1,abs(a).max()), vmax=max(1,abs(a).max()))
        axes[0].set(xlabel="Decision",ylabel="State dimension",title="Learned state · no semantic labels")
        axes[1].plot(np.linalg.norm(a,axis=1),color="#60e1c1", label="L2 norm")
        axes[1].set(xlabel="Decision",title="State magnitude")
    else:
        axes[0].text(.1,.5,"No retained state",color=FG)
    fig.tight_layout()
    return fig

def trajectory_figure(timeline):
    fig, axes = plt.subplots(1,2,figsize=(9,3))
    for ax in axes:
        style(ax)
    a = np.asarray([r["hidden"] for r in timeline if r["hidden"]])
    if len(a) > 1:
        centered = a-a.mean(axis=0)
        u,s,_ = np.linalg.svd(centered,full_matrices=False)
        z = u[:,:2]*s[:2]
        axes[0].scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap="viridis",s=12)
        axes[0].plot(z[:,0],z[:,1],alpha=.3,color="#60e1c1")
        axes[0].set(title="Episode PCA · descriptive only",xlabel="PC1",ylabel="PC2")
        cosine = (a[1:]*a[:-1]).sum(axis=1)/(np.linalg.norm(a[1:],axis=1)*np.linalg.norm(a[:-1],axis=1)+1e-8)
        axes[1].plot(cosine,color="#bc9565")
        axes[1].set(title="Consecutive state cosine",xlabel="Decision",ylim=(-1.1,1.1))
    fig.tight_layout()
    return fig
