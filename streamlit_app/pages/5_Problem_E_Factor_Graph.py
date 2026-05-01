"""
Problem E — Contributing Factor Graph
=======================================
Three tabs:
  1. Network View    — interactive Plotly scatter graph, colored by node type / community
  2. Centrality      — ranked table + bar charts; filter by node type
  3. Factor Patterns — frequent factor co-occurrence patterns with severity and lift
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from streamlit_app.utils.loaders import load_graph_data, load_factor_patterns

st.set_page_config(page_title="Problem E — Factor Graph", page_icon="🔵", layout="wide")
st.title("🔵 Problem E — Contributing Factor Knowledge Graph")
st.caption(
    "Co-occurrence graph of contributing factors, flight phases, FAR parts, and aircraft types. "
    "Louvain community detection · betweenness centrality · high-severity critical paths."
)

try:
    G, centrality = load_graph_data()
    patterns = load_factor_patterns()
except Exception as e:
    st.error(f"Could not load graph data: {e}")
    st.info("Run `scripts/run_phase5.py` to generate the required artefacts.")
    st.stop()

import networkx as nx

tab1, tab2, tab3 = st.tabs(["🕸️ Network View", "📊 Centrality Analysis", "🔗 Factor Patterns"])

# ── shared color maps ──────────────────────────────────────────
NODE_TYPE_COLORS = {
    "FACTOR":   "#e74c3c",
    "PHASE":    "#3498db",
    "AIRCRAFT": "#2ecc71",
    "FAR":      "#f39c12",
}
COMMUNITY_PALETTE = px.colors.qualitative.Plotly


# ──────────────────────────────────────────────────────────────
# TAB 1 — Network View
# ──────────────────────────────────────────────────────────────

with tab1:
    col_opts, col_info = st.columns([1, 2])

    with col_opts:
        color_by = st.radio(
            "Color nodes by",
            ["Node Type", "Community", "Avg Severity"],
            horizontal=False,
            key="color_by",
        )
        min_weight = st.slider(
            "Min edge weight (co-occurrence count)",
            min_value=10, max_value=2000, value=200, step=50,
            key="min_weight",
        )
        show_labels = st.checkbox("Show node labels", value=True, key="show_labels")

    with col_info:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_communities = len(set(nx.get_node_attributes(G, "community").values()))
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", n_nodes)
        c2.metric("Edges (all)", n_edges)
        c3.metric("Communities", n_communities)
        st.caption(
            "Nodes: **FACTOR** (contributing factors), **PHASE** (flight phases), "
            "**AIRCRAFT** (aircraft types), **FAR** (regulatory parts). "
            "Edge weight = co-occurrence count in ASRS reports."
        )

    # Filter edges by weight
    edges_filtered = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) >= min_weight]
    nodes_in_subgraph = set()
    for u, v, _ in edges_filtered:
        nodes_in_subgraph.add(u)
        nodes_in_subgraph.add(v)

    H = G.subgraph(nodes_in_subgraph)

    # Layout
    pos = nx.spring_layout(H, seed=42, k=1.5 / max(1, H.number_of_nodes() ** 0.5))

    # Build edge traces (bundle all edges into one trace for performance)
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#cccccc"),
        hoverinfo="none",
        showlegend=False,
    )

    # Build node traces (one per type for legend)
    node_types = sorted(set(nx.get_node_attributes(H, "node_type").values()))
    node_traces = []

    for nt in node_types:
        nodes_of_type = [n for n, d in H.nodes(data=True) if d.get("node_type") == nt]
        if not nodes_of_type:
            continue

        nx_vals, ny_vals, texts, hover_texts, sizes, colors_arr = [], [], [], [], [], []

        for n in nodes_of_type:
            x, y = pos[n]
            d = H.nodes[n]
            count = d.get("count", 1)
            avg_sev = d.get("avg_severity", 0)
            community = d.get("community", 0)
            short_name = n.split(":", 1)[1] if ":" in n else n

            nx_vals.append(x)
            ny_vals.append(y)
            texts.append(short_name if show_labels else "")
            hover_texts.append(
                f"<b>{short_name}</b><br>"
                f"Type: {nt}<br>"
                f"Reports: {count:,}<br>"
                f"Avg Severity: {avg_sev:.3f}<br>"
                f"Community: {community}"
            )
            sizes.append(max(8, min(35, np.log1p(count) * 3)))

            if color_by == "Node Type":
                colors_arr.append(NODE_TYPE_COLORS.get(nt, "#999"))
            elif color_by == "Community":
                colors_arr.append(COMMUNITY_PALETTE[community % len(COMMUNITY_PALETTE)])
            else:  # Avg Severity
                colors_arr.append(avg_sev)

        if color_by == "Avg Severity":
            node_traces.append(go.Scatter(
                x=nx_vals, y=ny_vals, mode="markers+text" if show_labels else "markers",
                text=texts, textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=sizes,
                    color=colors_arr,
                    colorscale="RdYlGn_r",
                    cmin=0, cmax=2,
                    colorbar=dict(title="Avg Severity") if nt == node_types[0] else None,
                    line=dict(width=1, color="white"),
                ),
                hovertext=hover_texts, hoverinfo="text",
                name=nt,
            ))
        else:
            node_traces.append(go.Scatter(
                x=nx_vals, y=ny_vals, mode="markers+text" if show_labels else "markers",
                text=texts, textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=sizes,
                    color=colors_arr,
                    line=dict(width=1, color="white"),
                ),
                hovertext=hover_texts, hoverinfo="text",
                name=nt,
            ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=f"Factor Co-occurrence Graph (edges ≥ {min_weight} co-occurrences)",
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(
            title="Node Type",
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Node size ∝ log(report count). "
        "Drag to explore. Hover for details. "
        "Increase the edge weight threshold to reduce clutter."
    )


# ──────────────────────────────────────────────────────────────
# TAB 2 — Centrality Analysis
# ──────────────────────────────────────────────────────────────

with tab2:
    st.markdown(
        "Nodes ranked by **betweenness centrality** — how often a node lies on the shortest "
        "path between other nodes. High betweenness = critical connector / bottleneck."
    )

    type_filter = st.multiselect(
        "Filter by node type",
        options=sorted(centrality["type"].unique()),
        default=sorted(centrality["type"].unique()),
        key="cent_type_filter",
    )
    cent_filtered = centrality[centrality["type"].isin(type_filter)].copy()
    cent_filtered = cent_filtered.sort_values("betweenness", ascending=False).reset_index(drop=True)
    cent_filtered["short_name"] = cent_filtered["node"].str.split(":").str[-1]

    top_n_cent = st.slider("Show top N nodes", 10, 60, 20, key="cent_top_n")

    col_bc, col_dc = st.columns(2)

    with col_bc:
        df_bc = cent_filtered.head(top_n_cent)
        fig = px.bar(
            df_bc,
            x="betweenness",
            y="short_name",
            orientation="h",
            color="type",
            color_discrete_map=NODE_TYPE_COLORS,
            hover_data={"count": True, "avg_severity": ":.3f"},
            title="Betweenness Centrality (top nodes)",
            labels={"betweenness": "Betweenness", "short_name": "Node"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=max(350, top_n_cent * 22), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_dc:
        df_wt = cent_filtered.sort_values("weighted_degree", ascending=False).head(top_n_cent)
        fig2 = px.bar(
            df_wt,
            x="weighted_degree",
            y="short_name",
            orientation="h",
            color="type",
            color_discrete_map=NODE_TYPE_COLORS,
            hover_data={"count": True, "avg_severity": ":.3f"},
            title="Weighted Degree (co-occurrence volume)",
            labels={"weighted_degree": "Weighted Degree", "short_name": "Node"},
        )
        fig2.update_yaxes(autorange="reversed")
        fig2.update_layout(height=max(350, top_n_cent * 22), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Severity vs Centrality scatter
    st.subheader("Severity vs. Betweenness")
    fig3 = px.scatter(
        cent_filtered,
        x="betweenness",
        y="avg_severity",
        size="count",
        color="type",
        color_discrete_map=NODE_TYPE_COLORS,
        hover_name="short_name",
        hover_data={"count": True},
        title="Node Avg Severity vs. Betweenness Centrality (bubble = report count)",
        labels={"betweenness": "Betweenness", "avg_severity": "Avg Severity"},
    )
    fig3.add_hline(y=float(cent_filtered["avg_severity"].mean()),
                   line_dash="dash", line_color="gray", annotation_text="Mean severity")
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Full table
    st.subheader("Full Centrality Table")
    display_cent = cent_filtered[["short_name", "type", "count", "avg_severity",
                                   "degree_centrality", "betweenness", "weighted_degree"]].copy()
    display_cent.columns = ["Node", "Type", "Reports", "Avg Severity",
                             "Degree Centrality", "Betweenness", "Weighted Degree"]
    display_cent["Avg Severity"] = display_cent["Avg Severity"].round(3)
    display_cent["Degree Centrality"] = display_cent["Degree Centrality"].round(4)
    display_cent["Betweenness"] = display_cent["Betweenness"].round(4)
    st.dataframe(display_cent.set_index("Node"), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 — Factor Patterns
# ──────────────────────────────────────────────────────────────

with tab3:
    st.markdown(
        "Frequent **factor co-occurrence patterns** mined from the incident graph. "
        "`Support` = number of incidents containing all factors in the pattern. "
        "`Lift` = how much more often this combination appears than expected by chance."
    )

    if not patterns:
        st.info("No factor patterns found. Ensure `scripts/run_phase5.py` has been executed.")
    else:
        df_pat = pd.DataFrame(patterns)

        # Format factors list for display
        df_pat["factors_str"] = df_pat["factors"].apply(
            lambda f: " + ".join(f) if isinstance(f, list) else str(f)
        )

        sort_col = st.selectbox(
            "Sort patterns by",
            ["Risk Score (Severity × Lift)", "Support", "Avg Severity", "Lift"],
            key="pat_sort",
        )
        df_pat["risk_proxy"] = df_pat["avg_severity"] * df_pat["lift"]

        sort_map = {
            "Risk Score (Severity × Lift)": "risk_proxy",
            "Support": "support",
            "Avg Severity": "avg_severity",
            "Lift": "lift",
        }
        df_pat_sorted = df_pat.sort_values(sort_map[sort_col], ascending=False).reset_index(drop=True)

        # Bar chart
        fig = px.bar(
            df_pat_sorted.head(15),
            x="risk_proxy" if "Risk" in sort_col else sort_map[sort_col],
            y="factors_str",
            orientation="h",
            color="avg_severity",
            color_continuous_scale="RdYlGn_r",
            hover_data={"support": True, "lift": ":.2f", "avg_severity": ":.3f"},
            title=f"Top 15 Factor Patterns (sorted by {sort_col})",
            labels={"risk_proxy": "Severity × Lift", "factors_str": "Pattern",
                    "avg_severity": "Avg Severity"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=420, margin=dict(l=280))
        st.plotly_chart(fig, use_container_width=True)

        # Severity vs Lift scatter
        st.subheader("Severity vs. Lift")
        fig2 = px.scatter(
            df_pat_sorted,
            x="lift",
            y="avg_severity",
            size="support",
            color="risk_proxy",
            color_continuous_scale="RdYlGn_r",
            hover_name="factors_str",
            hover_data={"support": True},
            title="Factor Patterns: Severity vs. Lift (bubble = incident support)",
            labels={"lift": "Lift", "avg_severity": "Avg Severity", "risk_proxy": "Sev × Lift"},
        )
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Severity = 1")
        fig2.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Lift = 1")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.subheader("All Factor Patterns")
        display_pat = df_pat_sorted[["factors_str", "support", "avg_severity", "lift", "risk_proxy"]].copy()
        display_pat.columns = ["Factors", "Support", "Avg Severity", "Lift", "Sev × Lift"]
        display_pat["Avg Severity"] = display_pat["Avg Severity"].round(3)
        display_pat["Lift"] = display_pat["Lift"].round(3)
        display_pat["Sev × Lift"] = display_pat["Sev × Lift"].round(3)
        st.dataframe(display_pat.set_index("Factors"), use_container_width=True)

        st.caption(
            "Lift > 1.5 and Avg Severity > 1.5 together indicate a **high-risk factor cluster** — "
            "a combination disproportionately associated with serious incidents."
        )
