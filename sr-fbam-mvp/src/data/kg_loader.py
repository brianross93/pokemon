"""
Knowledge Graph Data Loaders
Aligned with Overview.md:101-174 and generated data format
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx


@dataclass
class Node:
    """Knowledge graph node"""
    node_id: str
    node_type: str
    name: str
    aliases: List[str]
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "Node":
        aliases = row["aliases"].split("|") if row["aliases"] else []
        return cls(
            node_id=row["node_id"],
            node_type=row["node_type"],
            name=row["name"],
            aliases=aliases
        )


@dataclass
class Edge:
    """Knowledge graph edge"""
    src_id: str
    relation: str
    dst_id: str
    weight: float
    is_noisy: bool = False
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "Edge":
        return cls(
            src_id=row["src_id"],
            relation=row["relation"],
            dst_id=row["dst_id"],
            weight=float(row["weight"]),
            is_noisy=row.get("noisy", "false").lower() == "true"
        )


@dataclass
class Query:
    """Multi-hop query"""
    query_id: str
    natural_language: str
    symbolic_plan: List[Dict[str, str]]
    answer_id: str
    difficulty: str = "baseline"
    hop_count: int = 1
    
    @classmethod
    def from_jsonl(cls, line: str) -> "Query":
        data = json.loads(line)
        return cls(**data)


class KnowledgeGraph:
    """Wrapper around NetworkX graph with SR-FBAM utilities"""
    
    def __init__(self, nodes: List[Node], edges: List[Edge], relations: Dict):
        self.nodes_by_id: Dict[str, Node] = {n.node_id: n for n in nodes}
        self.edges = edges
        self.relations = relations
        
        # Build NetworkX graph
        self.graph = nx.MultiDiGraph()
        
        # Add nodes
        for node in nodes:
            self.graph.add_node(
                node.node_id,
                node_type=node.node_type,
                name=node.name,
                aliases=node.aliases
            )
        
        # Add edges
        for edge in edges:
            self.graph.add_edge(
                edge.src_id,
                edge.dst_id,
                relation=edge.relation,
                weight=edge.weight,
                is_noisy=edge.is_noisy
            )
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve node by ID"""
        return self.nodes_by_id.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes_by_id.values() if n.node_type == node_type]
    
    def get_nodes_by_name(self, name: str, fuzzy: bool = False) -> List[Node]:
        """Find nodes by name (exact or fuzzy match)"""
        if not fuzzy:
            return [n for n in self.nodes_by_id.values() if n.name == name or name in n.aliases]
        else:
            name_lower = name.lower()
            return [
                n for n in self.nodes_by_id.values()
                if name_lower in n.name.lower() or any(name_lower in a.lower() for a in n.aliases)
            ]
    
    def assoc(self, relation: str, target_id: Optional[str] = None, 
              source_id: Optional[str] = None, node_type: Optional[str] = None) -> List[str]:
        """
        ASSOC operation: find nodes connected by a relation
        - If target_id given: find all sources with relation -> target
        - If source_id given: find all targets with source -> relation
        - Optionally filter by node_type
        """
        results = []
        
        if target_id:
            # Find incoming edges to target
            for edge in self.edges:
                if edge.dst_id == target_id and edge.relation == relation:
                    node = self.get_node(edge.src_id)
                    if node is None:
                        continue
                    if node_type is None or node.node_type == node_type:
                        results.append(edge.src_id)
        
        elif source_id:
            # Find outgoing edges from source
            for edge in self.edges:
                if edge.src_id == source_id and edge.relation == relation:
                    node = self.get_node(edge.dst_id)
                    if node is None:
                        continue
                    if node_type is None or node.node_type == node_type:
                        results.append(edge.dst_id)
        
        return results
    
    def follow(self, source_id: str, relation: str, node_type: Optional[str] = None) -> List[str]:
        """
        FOLLOW operation: traverse from source along relation
        Same as assoc with source_id, but clearer semantics
        """
        return self.assoc(relation, source_id=source_id, node_type=node_type)
    
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        """Get all neighbors of a node, optionally filtered by relation"""
        neighbors = []
        for edge in self.edges:
            if edge.src_id == node_id:
                if relation is None or edge.relation == relation:
                    neighbors.append((edge.dst_id, edge.relation))
        return neighbors
    
    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes (ignores relations)"""
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes_by_id)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)


def load_kg(data_dir: Path, variant: Optional[str] = None) -> KnowledgeGraph:
    """
    Load knowledge graph from CSV files
    
    Args:
        data_dir: Base data directory
        variant: Optional stress test variant (e.g., "ambiguous_nodes", "noisy_edges")
    
    Returns:
        KnowledgeGraph object
    """
    # Determine which files to load
    if variant:
        variant_dir = data_dir / "stress_variants" / variant
        nodes_file = variant_dir / "nodes_override.csv" if (variant_dir / "nodes_override.csv").exists() else data_dir / "nodes.csv"
        edges_file = variant_dir / "edges_noisy.csv" if variant == "noisy_edges" else data_dir / "edges.csv"
        if variant == "long_chains" and (variant_dir / "edges_extended.csv").exists():
            edges_file = variant_dir / "edges_extended.csv"
    else:
        nodes_file = data_dir / "nodes.csv"
        edges_file = data_dir / "edges.csv"
    
    # Load nodes
    nodes = []
    with open(nodes_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(Node.from_csv_row(row))
    
    # Load edges
    edges = []
    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(Edge.from_csv_row(row))
    
    # Load relations metadata
    relations_file = data_dir / "relations.json"
    with open(relations_file, "r", encoding="utf-8") as f:
        relations = json.load(f)
    
    # Handle missing nodes variant
    if variant == "missing_nodes":
        removed_file = data_dir / "stress_variants" / "missing_nodes" / "removed_nodes.txt"
        with open(removed_file, "r", encoding="utf-8") as f:
            removed_ids = set(f.read().strip().split("\n"))
        nodes = [n for n in nodes if n.node_id not in removed_ids]
        edges = [e for e in edges if e.src_id not in removed_ids and e.dst_id not in removed_ids]
    
    return KnowledgeGraph(nodes, edges, relations)


def load_queries(query_file: Path) -> List[Query]:
    """Load queries from JSONL file"""
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(Query.from_jsonl(line))
    return queries


def load_metadata(data_dir: Path) -> Dict:
    """Load metadata about the knowledge graph"""
    metadata_file = data_dir / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_relation_info(data_dir: Path) -> Dict:
    """Load relation definitions"""
    relations_file = data_dir / "relations.json"
    with open(relations_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Convenience function for quick loading
def load_dataset(data_dir: Path, variant: Optional[str] = None, 
                 split: str = "train") -> Tuple[KnowledgeGraph, List[Query]]:
    """
    Load KG and queries in one call
    
    Args:
        data_dir: Base data directory
        variant: Optional stress test variant
        split: "train" or "eval"
    
    Returns:
        (KnowledgeGraph, List[Query])
    """
    kg = load_kg(data_dir, variant)
    query_file = data_dir / f"queries_{split}.jsonl"
    queries = load_queries(query_file)
    return kg, queries


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("Loading baseline KG...")
    kg = load_kg(data_dir)
    print(f"Loaded {kg.num_nodes} nodes, {kg.num_edges} edges")
    
    print("\nLoading train queries...")
    queries = load_queries(data_dir / "queries_train.jsonl")
    print(f"Loaded {len(queries)} queries")
    
    print("\nSample query:")
    q = queries[0]
    print(f"  ID: {q.query_id}")
    print(f"  Question: {q.natural_language}")
    print(f"  Hops: {q.hop_count}")
    print(f"  Answer: {q.answer_id}")
    
    print("\nTesting ASSOC operation...")
    # Find actors born in Paris
    paris_nodes = kg.get_nodes_by_name("Paris")
    if paris_nodes:
        paris_id = paris_nodes[0].node_id
        actors = kg.assoc("born_in", target_id=paris_id, node_type="Actor")
        print(f"  Found {len(actors)} actors born in Paris")
        if actors:
            actor = kg.get_node(actors[0])
            print(f"    Example: {actor.name}")
    
    print("\nTesting stress variant loading...")
    kg_noisy = load_kg(data_dir, variant="noisy_edges")
    print(f"  Noisy variant: {kg_noisy.num_edges} edges")
    noisy_count = sum(1 for e in kg_noisy.edges if e.is_noisy)
    print(f"  Noisy edges: {noisy_count}")

