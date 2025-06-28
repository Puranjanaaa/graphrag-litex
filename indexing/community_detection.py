"""
Community detection algorithms for GraphRAG using Infomap or Louvain.
"""
from typing import List, Dict, Any
import networkx as nx
from collections import defaultdict
import numpy as np

try:
    import infomap
    INFOMAP_AVAILABLE = True
except ImportError:
    INFOMAP_AVAILABLE = False
    try:
        import community as community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False

from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph


class CommunityDetector:
    """
    Detects communities using Infomap or Louvain and builds hierarchical structure.
    """

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.resolution = getattr(config, 'community_resolution', 1.0)
        self.min_community_size = getattr(config, 'min_community_size', 3)
        self.use_weights = getattr(config, 'use_edge_weights', True)
        self.directed = getattr(config, 'directed_graph', True)
        self.hierarchical_levels = getattr(config, 'hierarchical_levels', 3)
        self.num_trials = getattr(config, 'infomap_trials', 10)
        self.markov_time = getattr(config, 'markov_time', 1.0)

        if not INFOMAP_AVAILABLE:
            print("Warning: Infomap not available, falling back to Louvain.")

    def detect_communities(self, kg: KnowledgeGraph) -> Dict[str, Dict[str, Any]]:
        """
        Run hierarchical community detection and return structured community dict.
        """
        graph = kg.get_graph()

        if INFOMAP_AVAILABLE:
            communities = self._detect_with_infomap(graph, kg)
        elif LOUVAIN_AVAILABLE:
            communities = self._detect_with_louvain(graph)
        else:
            raise ImportError("Neither Infomap nor Louvain is available")

        communities = self._enforce_min_community_size(communities)
        hierarchical_communities = self._create_hierarchy(communities, graph)
        return hierarchical_communities

    def _detect_with_infomap(self, graph: nx.Graph, kg: KnowledgeGraph) -> Dict[str, List[str]]:
        infomap_args = ["--two-level", "--silent"]
        if self.directed and graph.is_directed():
            infomap_args.append("--directed")

        im = infomap.Infomap(" ".join(infomap_args))
        im.numTrials = self.num_trials
        im.markovTime = self.markov_time

        node_to_id = {node: i for i, node in enumerate(graph.nodes())}
        id_to_node = {i: node for node, i in node_to_id.items()}

        for node in graph.nodes():
            im.addNode(node_to_id[node])
        for source, target, edge_data in graph.edges(data=True):
            weight = float(edge_data.get('weight', 1.0))
            if self.use_weights and weight == 1.0:
                rel_key = f"{source}_{target}"
                if rel_key in kg.relationships:
                    weight = kg.relationships[rel_key].strength
                elif not self.directed:
                    rel_key_rev = f"{target}_{source}"
                    if rel_key_rev in kg.relationships:
                        weight = kg.relationships[rel_key_rev].strength
            im.addLink(node_to_id[source], node_to_id[target], weight)

        im.run()

        communities = defaultdict(list)
        for node in im.tree:
            if node.isLeaf:
                original_node = id_to_node[node.physicalId]
                communities[str(node.moduleIndex())].append(original_node)

        return dict(communities)

    def _detect_with_louvain(self, graph: nx.Graph) -> Dict[str, List[str]]:
        partition = community_louvain.best_partition(graph, resolution=self.resolution)
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[str(comm_id)].append(node)
        return dict(communities)

    def _enforce_min_community_size(self, communities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        small, valid = {}, {}
        for cid, nodes in communities.items():
            (valid if len(nodes) >= self.min_community_size else small)[cid] = nodes

        if not valid and small:
            largest = max(small, key=lambda k: len(small[k]))
            valid[largest] = small.pop(largest)

        if small:
            target = max(valid, key=lambda k: len(valid[k]))
            for nodes in small.values():
                valid[target].extend(nodes)

        return valid

    def _create_hierarchy(self, communities: Dict[str, List[str]], graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
        hierarchy = {
            "C0": {
                "level": 0,
                "communities": {}
            }
        }
        for cid, nodes in communities.items():
            subgraph = graph.subgraph(nodes)
            metrics = self._calculate_community_flow(nodes, graph)
            hierarchy["C0"]["communities"][cid] = {
                "nodes": nodes,
                "size": len(nodes),
                "density": nx.density(subgraph) if len(nodes) > 1 else 0.0,
                **metrics,
                "sub_communities": {}
            }

        self._create_sub_levels_infomap(hierarchy, graph, 0, self.hierarchical_levels)
        return hierarchy

    def _calculate_community_flow(self, nodes: List[str], graph: nx.Graph) -> Dict[str, float]:
        if len(nodes) <= 1:
            return {"flow": 0.0, "description_length": 0.0}

        sub = graph.subgraph(nodes)
        internal = sub.number_of_edges()
        external = sum(1 for n in nodes for nb in graph.neighbors(n) if nb not in nodes)
        total = internal + external

        if total > 0:
            pi = internal / total
            pe = external / total
            dl = -sum(p * np.log2(p) for p in (pi, pe) if p > 0)
        else:
            pi = dl = 0.0

        return {"flow": pi, "description_length": dl}

    def _create_sub_levels_infomap(self, hierarchy, graph, current_level, max_levels):
        if current_level >= max_levels - 1 or not INFOMAP_AVAILABLE:
            return

        next_level_key = f"C{current_level + 1}"
        hierarchy[next_level_key] = {
            "level": current_level + 1,
            "communities": {}
        }

        counter = 0
        for cid, cdata in hierarchy[f"C{current_level}"]["communities"].items():
            nodes = cdata["nodes"]
            if len(nodes) <= self.min_community_size * 2:
                next_id = f"{current_level+1}_{counter}"
                hierarchy[next_level_key]["communities"][next_id] = {
                    **cdata,
                    "parent": cid
                }
                cdata["sub_communities"][next_id] = len(nodes)
                counter += 1
                continue

            sub = graph.subgraph(nodes)
            sub_communities = self._detect_subgraph_communities(sub)

            for sub_nodes in sub_communities.values():
                if len(sub_nodes) >= self.min_community_size:
                    next_id = f"{current_level+1}_{counter}"
                    metrics = self._calculate_community_flow(sub_nodes, graph)
                    hierarchy[next_level_key]["communities"][next_id] = {
                        "nodes": sub_nodes,
                        "size": len(sub_nodes),
                        "density": nx.density(graph.subgraph(sub_nodes)) if len(sub_nodes) > 1 else 0.0,
                        **metrics,
                        "parent": cid,
                        "sub_communities": {}
                    }
                    cdata["sub_communities"][next_id] = len(sub_nodes)
                    counter += 1

        self._create_sub_levels_infomap(hierarchy, graph, current_level + 1, max_levels)

    def _detect_subgraph_communities(self, subgraph: nx.Graph) -> Dict[str, List[str]]:
        if not INFOMAP_AVAILABLE or len(subgraph) <= self.min_community_size:
            return {"0": list(subgraph.nodes())}

        im = infomap.Infomap("--two-level --silent")
        im.numTrials = min(self.num_trials, 5)
        node_to_id = {node: idx for idx, node in enumerate(subgraph.nodes())}
        id_to_node = {idx: node for node, idx in node_to_id.items()}

        for n in subgraph.nodes():
            im.addNode(node_to_id[n])
        for s, t, ed in subgraph.edges(data=True):
            im.addLink(node_to_id[s], node_to_id[t], ed.get("weight", 1.0))

        im.run()
        result = defaultdict(list)
        for node in im.tree:
            if node.isLeaf:
                result[str(node.moduleIndex())].append(id_to_node[node.physicalId])
        return dict(result)

    def get_community_entities(
        self,
        kg: KnowledgeGraph,
        community_id: str,
        hierarchical_communities: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        for level, data in hierarchical_communities.items():
            if community_id in data["communities"]:
                community_data = data["communities"][community_id]
                break
        else:
            return {}

        nodes = community_data["nodes"]
        entities = {eid: e.to_dict() for eid, e in kg.entities.items() if eid in nodes}
        relationships = {
            rid: r.to_dict() for rid, r in kg.relationships.items()
            if r.source_id in nodes and r.target_id in nodes
        }
        claims = {
            cid: c.to_dict() for cid, c in kg.claims.items()
            if any(eid in nodes for eid in c.entity_ids)
        }

        return {
            "community_id": community_id,
            "level": data["level"],
            "size": len(nodes),
            "flow": community_data.get("flow", 0.0),
            "description_length": community_data.get("description_length", 0.0),
            "entities": entities,
            "relationships": relationships,
            "claims": claims
        }


def detect_communities(documents_path: str) -> List[List[str]]:
    """
    Standalone function for detecting communities from a documents path.
    This is a simplified implementation for compatibility with map_reduce.py
    """
    # For now, return a placeholder - this would need to be implemented
    # based on the specific requirements of the map_reduce processor
    return [["placeholder_community"]]
