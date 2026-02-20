"""
Canonical hashing for AST graphs using Weisfeiler-Leman style hashing.

This provides a fast way to identify potentially isomorphic graphs before
running the full isomorphism check.
"""

from collections.abc import Sequence

import networkx as nx
import xxhash

from .ast_to_graph import compute_label_histogram


def weisfeiler_leman_hash(graph: nx.DiGraph, iterations: int = 3) -> str:
    """
    Compute a Weisfeiler-Leman style hash for a directed labeled graph.

    This hash is the same for isomorphic graphs (but non-isomorphic graphs
    may also have the same hash - it's a necessary but not sufficient condition).

    Args:
        graph: A NetworkX directed graph with 'label' attribute on nodes
        iterations: Number of WL refinement iterations

    Returns:
        A hex string hash of the graph structure

    Complexity: O(k × (n + m × log m)) where k = iterations, n = nodes, m = edges
    Optimized: Uses integer labels and direct tuple hashing for efficiency
    """
    if graph.number_of_nodes() == 0:
        return "empty"

    # Initialize labels as integers - hash string labels to get consistent integers
    # This preserves label semantics across different graphs
    labels: dict[int, int] = {}
    for node, data in graph.nodes(data=True):
        label_str = data.get("label", "Unknown")
        # Hash the string label to get a consistent integer across graphs
        labels[node] = xxhash.xxh64(label_str.encode()).intdigest()

    # WL iterations: refine labels based on neighbor structure
    for _ in range(iterations):
        new_labels: dict[int, int] = {}
        for node in graph.nodes():
            # Tuple of (own_label, sorted_predecessor_labels, sorted_successor_labels)
            # Using tuples avoids string allocation overhead
            pred_labels = tuple(sorted(labels[p] for p in graph.predecessors(node)))
            succ_labels = tuple(sorted(labels[s] for s in graph.successors(node)))
            combined = (labels[node], pred_labels, succ_labels)
            # Hash tuple directly - intdigest() returns int, avoiding hex string overhead
            new_labels[node] = xxhash.xxh64(repr(combined).encode()).intdigest()

        labels = new_labels

    # Final hash: sorted multiset of all node labels
    sorted_labels = tuple(sorted(labels.values()))
    return xxhash.xxh64(repr(sorted_labels).encode()).hexdigest()


def structural_fingerprint(graph: nx.DiGraph) -> dict:
    """
    Compute a structural fingerprint for quick filtering before isomorphism check.

    Returns a dict of structural features that must match for graphs to be isomorphic.
    """
    if graph.number_of_nodes() == 0:
        return {"empty": True}

    # Basic counts
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # Label histogram
    label_counts = compute_label_histogram(graph)

    # In/out degree sequences (sorted)
    in_degrees = sorted(graph.in_degree(n) for n in graph.nodes())
    out_degrees = sorted(graph.out_degree(n) for n in graph.nodes())

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "label_counts": label_counts,
        "in_degree_seq": in_degrees,
        "out_degree_seq": out_degrees,
    }


def fingerprints_compatible(fp1: dict, fp2: dict) -> bool:
    """Check if two fingerprints are compatible (necessary for isomorphism)."""
    if fp1.get("empty") or fp2.get("empty"):
        return bool(fp1.get("empty") == fp2.get("empty"))

    return bool(
        fp1["n_nodes"] == fp2["n_nodes"]
        and fp1["n_edges"] == fp2["n_edges"]
        and fp1["label_counts"] == fp2["label_counts"]
        and fp1["in_degree_seq"] == fp2["in_degree_seq"]
        and fp1["out_degree_seq"] == fp2["out_degree_seq"]
    )


def compute_hierarchy_hash(graph: nx.DiGraph, max_depth: int = 5) -> Sequence[str]:
    """
    Compute hierarchical hashes at different depths.

    This allows for approximate matching - two graphs might match at depth 2
    but differ at depth 5, indicating they share high-level structure.

    Optimized: Single BFS pass instead of O(depth) BFS calls.
    Complexity: O(n + m) instead of O(depth × (n + m))
    """
    if graph.number_of_nodes() == 0:
        return ["empty"] * max_depth

    # Find roots (nodes with no incoming edges)
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
        roots = [0] if 0 in graph.nodes() else [next(iter(graph.nodes()))]

    # Single BFS pass: compute depth for all reachable nodes from all roots
    # node_depths[node] = minimum depth from any root
    node_depths: dict[int, int] = {}
    for root in roots:
        for node, d in nx.single_source_shortest_path_length(graph, root).items():
            if node not in node_depths or d < node_depths[node]:
                node_depths[node] = d

    # Pre-compute nodes at each depth level using cumulative sets
    # nodes_up_to_depth[d] = nodes with depth < d (reachable within d-1 hops)
    nodes_by_depth: list[set[int]] = [set() for _ in range(max_depth + 1)]
    for node, d in node_depths.items():
        # Add node to all depth levels greater than d
        for level in range(d + 1, max_depth + 1):
            nodes_by_depth[level].add(node)

    # Compute hashes for each depth level
    hashes: list[str] = []
    for depth in range(1, max_depth + 1):
        nodes_at_depth = nodes_by_depth[depth]
        if nodes_at_depth:
            subgraph = graph.subgraph(nodes_at_depth)
            h = weisfeiler_leman_hash(subgraph, iterations=2)
        else:
            h = "empty"
        hashes.append(h)

    return hashes
