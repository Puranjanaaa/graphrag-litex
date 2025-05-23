"""
Community detection algorithms for GraphRAG using Infomap.
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
    # Fallback to Louvain if Infomap is not available
    try:
        import community as community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False

from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph

class CommunityDetector:
    """
    Detects communities in the knowledge graph using Infomap algorithm.
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the community detector.
        
        Args:
            config: The GraphRAG configuration
        """
        self.config = config
        self.resolution = getattr(config, 'community_resolution', 1.0)
        self.min_community_size = getattr(config, 'min_community_size', 3)
        self.use_weights = getattr(config, 'use_edge_weights', True)
        self.directed = getattr(config, 'directed_graph', True)
        self.hierarchical_levels = getattr(config, 'hierarchical_levels', 3)
        
        # Infomap specific parameters
        self.num_trials = getattr(config, 'infomap_trials', 10)
        self.markov_time = getattr(config, 'markov_time', 1.0)
        
        if not INFOMAP_AVAILABLE:
            print("Warning: Infomap not available, falling back to Louvain algorithm")
    
    def detect_communities(self, kg: KnowledgeGraph) -> Dict[str, Dict[str, Any]]:
        """
        Detect communities in the knowledge graph using Infomap.
        
        Args:
            kg: The knowledge graph
            
        Returns:
            Dictionary mapping community IDs to community information
        """
        # Get the NetworkX graph
        graph = kg.get_graph()
        
        if INFOMAP_AVAILABLE:
            # Detect communities using Infomap algorithm
            communities = self._detect_with_infomap(graph, kg)
        elif LOUVAIN_AVAILABLE:
            # Fallback to Louvain
            communities = self._detect_with_louvain(graph)
        else:
            raise ImportError("Neither Infomap nor python-louvain is available")
        
        # Ensure minimum community size
        communities = self._enforce_min_community_size(communities)
        
        # Create a hierarchical structure
        hierarchical_communities = self._create_hierarchy(communities, graph)
        
        return hierarchical_communities
    
    def _detect_with_infomap(self, graph: nx.Graph, kg: KnowledgeGraph) -> Dict[str, List[str]]:
        """
        Detect communities using the Infomap algorithm.
        
        Args:
            graph: The NetworkX graph
            kg: The knowledge graph (for accessing relationship weights)
            
        Returns:
            Dictionary mapping community IDs to lists of node IDs
        """
        if len(graph) == 0:
            return {}
        
        # Create Infomap instance
        infomap_args = []
        
        # Configure for directed or undirected graph
        if self.directed and graph.is_directed():
            infomap_args.append("--directed")
        
        # Add other parameters
        infomap_args.extend([
            "--two-level",  # Enable hierarchical detection
            "--silent"  # Suppress output
        ])
        
        im = infomap.Infomap(" ".join(infomap_args))
        
        # Set parameters directly on the Infomap object
        if hasattr(im, 'numTrials'):
            im.numTrials = self.num_trials
        if hasattr(im, 'markovTime'):
            im.markovTime = self.markov_time
        
        # Create node ID mapping
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        
        # Add nodes to Infomap
        for node in graph.nodes():
            im.addNode(node_to_id[node])
        
        # Add edges with weights
        for source, target, edge_data in graph.edges(data=True):
            source_id = node_to_id[source]
            target_id = node_to_id[target]
            
            # Get weight from edge data or relationship
            weight = 1.0
            if self.use_weights and 'weight' in edge_data:
                weight = float(edge_data['weight'])
            elif self.use_weights:
                # Try to get weight from knowledge graph relationships
                rel_key = f"{source}_{target}"
                if rel_key in kg.relationships:
                    weight = kg.relationships[rel_key].strength
                # Try reverse direction for undirected graphs
                elif not self.directed:
                    rel_key_reverse = f"{target}_{source}"
                    if rel_key_reverse in kg.relationships:
                        weight = kg.relationships[rel_key_reverse].strength
            
            im.addLink(source_id, target_id, weight)
        
        # Run the algorithm
        im.run()
        
        # Extract communities
        communities = defaultdict(list)
        
        for node in im.tree:
            if node.isLeaf:
                node_id = node.physicalId
                module_id = node.moduleIndex()
                original_node = id_to_node[node_id]
                communities[str(module_id)].append(original_node)
        
        return dict(communities)
    
    def _detect_with_louvain(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """
        Fallback: Detect communities using the Louvain algorithm.
        
        Args:
            graph: The NetworkX graph
            
        Returns:
            Dictionary mapping community IDs to lists of node IDs
        """
        if len(graph) == 0:
            return {}
            
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(graph, resolution=self.resolution)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[str(community_id)].append(node)
        
        return dict(communities)
    
    def _detect_hierarchical_infomap(self, graph: nx.Graph, kg: KnowledgeGraph) -> Dict[int, Dict[str, List[str]]]:
        """
        Detect hierarchical communities using Infomap's multi-level detection.
        
        Args:
            graph: The NetworkX graph
            kg: The knowledge graph
            
        Returns:
            Dictionary mapping levels to community structures
        """
        if not INFOMAP_AVAILABLE or len(graph) == 0:
            return {}
        
        # Configure Infomap for hierarchical detection
        infomap_args = [
            "--tree",  # Enable hierarchical structure
            "--silent"
        ]
        
        if self.directed and graph.is_directed():
            infomap_args.append("--directed")
        
        im = infomap.Infomap(" ".join(infomap_args))
        
        # Set parameters directly on the Infomap object
        if hasattr(im, 'numTrials'):
            im.numTrials = self.num_trials
        
        # Create node mapping
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        
        # Add nodes and edges
        for node in graph.nodes():
            im.addNode(node_to_id[node])
        
        for source, target, edge_data in graph.edges(data=True):
            source_id = node_to_id[source]
            target_id = node_to_id[target]
            
            weight = 1.0
            if self.use_weights and 'weight' in edge_data:
                weight = float(edge_data['weight'])
            elif self.use_weights:
                rel_key = f"{source}_{target}"
                if rel_key in kg.relationships:
                    weight = kg.relationships[rel_key].strength
                elif not self.directed:
                    rel_key_reverse = f"{target}_{source}"
                    if rel_key_reverse in kg.relationships:
                        weight = kg.relationships[rel_key_reverse].strength
            
            im.addLink(source_id, target_id, weight)
        
        # Run algorithm
        im.run()
        
        # Extract hierarchical structure
        hierarchical_communities = defaultdict(lambda: defaultdict(list))
        
        for node in im.tree:
            if node.isLeaf:
                node_id = node.physicalId
                original_node = id_to_node[node_id]
                
                # Build path from root to leaf
                path = []
                current = node
                while current is not None:
                    if hasattr(current, 'moduleIndex'):
                        path.append(current.moduleIndex())
                    current = current.parent
                
                path.reverse()
                
                # Add to appropriate levels
                for level, module_id in enumerate(path):
                    if level < self.hierarchical_levels:
                        hierarchical_communities[level][str(module_id)].append(original_node)
        
        return dict(hierarchical_communities)
    
    def _enforce_min_community_size(self, communities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Ensure all communities meet the minimum size requirement.
        
        Args:
            communities: Dictionary mapping community IDs to lists of node IDs
            
        Returns:
            Updated communities with small ones merged
        """
        # Find small communities
        small_communities = {}
        valid_communities = {}
        
        for community_id, nodes in communities.items():
            if len(nodes) < self.min_community_size:
                small_communities[community_id] = nodes
            else:
                valid_communities[community_id] = nodes
        
        # If all communities are small, keep the largest one
        if not valid_communities and small_communities:
            largest_id = max(small_communities, key=lambda k: len(small_communities[k]))
            valid_communities[largest_id] = small_communities[largest_id]
            del small_communities[largest_id]
        
        # Merge small communities into the nearest valid community
        if small_communities and valid_communities:
            largest_id = max(valid_communities, key=lambda k: len(valid_communities[k]))
            for nodes in small_communities.values():
                valid_communities[largest_id].extend(nodes)
        
        return valid_communities
    
    def _create_hierarchy(
        self, 
        communities: Dict[str, List[str]], 
        graph: nx.Graph
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a hierarchical community structure.
        
        Args:
            communities: Dictionary mapping community IDs to lists of node IDs
            graph: The original NetworkX graph
            
        Returns:
            Hierarchical community structure
        """
        hierarchical_communities = {}
        
        # Create the root level (C0)
        root_level = {
            "level": 0,
            "communities": {}
        }
        
        for community_id, nodes in communities.items():
            community_graph = graph.subgraph(nodes)
            
            # Calculate additional Infomap-specific metrics
            flow_info = self._calculate_community_flow(nodes, graph)
            
            root_level["communities"][community_id] = {
                "nodes": nodes,
                "size": len(nodes),
                "density": nx.density(community_graph) if len(nodes) > 1 else 0.0,
                "flow": flow_info.get("flow", 0.0),
                "description_length": flow_info.get("description_length", 0.0),
                "sub_communities": {}
            }
        
        hierarchical_communities["C0"] = root_level
        
        # Create sub-levels recursively using Infomap on subgraphs
        self._create_sub_levels_infomap(hierarchical_communities, graph, 0, self.hierarchical_levels)
        
        return hierarchical_communities
    
    def _calculate_community_flow(self, nodes: List[str], graph: nx.Graph) -> Dict[str, float]:
        """
        Calculate flow-based metrics for a community (Infomap-inspired).
        
        Args:
            nodes: List of node IDs in the community
            graph: The graph
            
        Returns:
            Dictionary with flow metrics
        """
        if len(nodes) <= 1:
            return {"flow": 0.0, "description_length": 0.0}
        
        subgraph = graph.subgraph(nodes)
        
        # Calculate internal flow (proxy: internal degree sum)
        internal_edges = subgraph.number_of_edges()
        total_degree = sum(dict(subgraph.degree()).values())
        
        # Calculate external flow (edges leaving the community)
        external_edges = 0
        for node in nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes:
                    external_edges += 1
        
        # Flow metrics (simplified)
        total_flow = internal_edges + external_edges
        internal_flow = internal_edges / max(total_flow, 1)
        
        # Description length (simplified information-theoretic measure)
        if total_flow > 0:
            p_internal = internal_edges / total_flow
            p_external = external_edges / total_flow
            
            description_length = 0
            if p_internal > 0:
                description_length -= p_internal * np.log2(p_internal)
            if p_external > 0:
                description_length -= p_external * np.log2(p_external)
        else:
            description_length = 0
        
        return {
            "flow": internal_flow,
            "description_length": description_length,
            "internal_edges": internal_edges,
            "external_edges": external_edges
        }
    
    def _create_sub_levels_infomap(
        self, 
        hierarchical_communities: Dict[str, Dict[str, Any]], 
        graph: nx.Graph, 
        current_level: int, 
        max_levels: int
    ):
        """
        Recursively create sub-levels using Infomap on subgraphs.
        
        Args:
            hierarchical_communities: Hierarchical community structure to update
            graph: The original NetworkX graph
            current_level: Current level in the hierarchy
            max_levels: Maximum number of levels
        """
        if current_level >= max_levels - 1 or not INFOMAP_AVAILABLE:
            return
        
        next_level = current_level + 1
        level_key = f"C{next_level}"
        
        # Create the next level
        hierarchical_communities[level_key] = {
            "level": next_level,
            "communities": {}
        }
        
        # Process each community at the current level
        current_level_key = f"C{current_level}"
        current_communities = hierarchical_communities[current_level_key]["communities"]
        
        community_counter = 0
        
        for community_id, community_data in current_communities.items():
            nodes = community_data["nodes"]
            
            # If the community is too small, don't subdivide
            if len(nodes) <= self.min_community_size * 2:
                # Project this community to the next level without subdivision
                next_community_id = f"{next_level}_{community_counter}"
                community_counter += 1
                
                hierarchical_communities[level_key]["communities"][next_community_id] = {
                    "nodes": nodes,
                    "size": len(nodes),
                    "density": community_data["density"],
                    "flow": community_data.get("flow", 0.0),
                    "description_length": community_data.get("description_length", 0.0),
                    "parent": community_id,
                    "sub_communities": {}
                }
                
                # Update the current level to reference this child
                current_communities[community_id]["sub_communities"][next_community_id] = len(nodes)
                continue
            
            # Extract the subgraph for this community
            subgraph = graph.subgraph(nodes)
            
            # Run Infomap on the subgraph
            sub_communities = self._detect_subgraph_communities(subgraph)
            
            # Add sub-communities to the next level
            for sub_nodes in sub_communities.values():
                if len(sub_nodes) >= self.min_community_size:
                    next_community_id = f"{next_level}_{community_counter}"
                    community_counter += 1
                    
                    sub_subgraph = subgraph.subgraph(sub_nodes)
                    flow_info = self._calculate_community_flow(sub_nodes, graph)
                    
                    hierarchical_communities[level_key]["communities"][next_community_id] = {
                        "nodes": sub_nodes,
                        "size": len(sub_nodes),
                        "density": nx.density(sub_subgraph) if len(sub_nodes) > 1 else 0.0,
                        "flow": flow_info.get("flow", 0.0),
                        "description_length": flow_info.get("description_length", 0.0),
                        "parent": community_id,
                        "sub_communities": {}
                    }
                    
                    # Update the current level to reference this child
                    current_communities[community_id]["sub_communities"][next_community_id] = len(sub_nodes)
        
        # Recursively process the next level
        self._create_sub_levels_infomap(hierarchical_communities, graph, next_level, max_levels)
    
    def _detect_subgraph_communities(self, subgraph: nx.Graph) -> Dict[str, List[str]]:
        """
        Detect communities in a subgraph using Infomap.
        
        Args:
            subgraph: The subgraph to analyze
            
        Returns:
            Dictionary mapping community IDs to node lists
        """
        if not INFOMAP_AVAILABLE or len(subgraph) <= self.min_community_size:
            # Return single community containing all nodes
            return {"0": list(subgraph.nodes())}
        
        # Configure Infomap for subgraph
        im = infomap.Infomap("--two-level --silent")
        
        # Set parameters directly on the Infomap object
        if hasattr(im, 'numTrials'):
            im.numTrials = min(self.num_trials, 5)  # Fewer trials for subgraphs
        
        # Create node mapping for subgraph
        node_to_id = {node: idx for idx, node in enumerate(subgraph.nodes())}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        
        # Add nodes and edges
        for node in subgraph.nodes():
            im.addNode(node_to_id[node])
        
        for source, target, edge_data in subgraph.edges(data=True):
            source_id = node_to_id[source]
            target_id = node_to_id[target]
            weight = edge_data.get('weight', 1.0)
            im.addLink(source_id, target_id, weight)
        
        # Run algorithm
        im.run()
        
        # Extract communities
        communities = defaultdict(list)
        for node in im.tree:
            if node.isLeaf:
                node_id = node.physicalId
                module_id = node.moduleIndex()
                original_node = id_to_node[node_id]
                communities[str(module_id)].append(original_node)
        
        return dict(communities)
    
    def get_community_entities(
        self, 
        kg: KnowledgeGraph, 
        community_id: str, 
        hierarchical_communities: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get the entities and relationships for a community.
        
        Args:
            kg: The knowledge graph
            community_id: The community ID
            hierarchical_communities: The hierarchical community structure
            
        Returns:
            Dictionary with entity and relationship data for the community
        """
        # Find the community in the hierarchy
        community_data = None
        community_level = None
        
        for level_key, level_data in hierarchical_communities.items():
            if community_id in level_data["communities"]:
                community_data = level_data["communities"][community_id]
                community_level = level_data["level"]
                break
        
        if not community_data:
            return {}
        
        # Get the entity nodes
        node_ids = community_data["nodes"]
        
        # Extract entities
        entities = {}
        for node_id in node_ids:
            if node_id in kg.entities:
                entities[node_id] = kg.entities[node_id].to_dict()
        
        # Extract relationships within this community
        relationships = {}
        for rel_id, rel in kg.relationships.items():
            if rel.source_id in node_ids and rel.target_id in node_ids:
                relationships[rel_id] = rel.to_dict()
        
        # Extract claims for these entities
        claims = {}
        for claim_id, claim in kg.claims.items():
            # Check if any entity in the claim is in this community
            if any(entity_id in node_ids for entity_id in claim.entity_ids):
                claims[claim_id] = claim.to_dict()
        
        return {
            "community_id": community_id,
            "level": community_level,
            "size": len(node_ids),
            "flow": community_data.get("flow", 0.0),
            "description_length": community_data.get("description_length", 0.0),
            "entities": entities,
            "relationships": relationships,
            "claims": claims
        }