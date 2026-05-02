"""
@module    GraphRAG
@description Graph RAG — builds a knowledge graph from document chunks
             and retrieves connected concepts for richer context.
             Instead of just finding similar chunks, it traverses
             relationships: Newton → Laws of Motion → Force → Engineering
             Gives LLM connected knowledge, not just similar text.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from .base_rag import BaseRAG, RAGQuery, RAGResult


class GraphRAG(BaseRAG):
    """
    Graph RAG — relationship-aware retrieval pipeline.

    Key difference from all previous RAGs:
      - Builds an in-memory knowledge graph from chunks
      - Nodes = concepts (key terms extracted from chunks)
      - Edges = co-occurrence (concepts appearing in same chunk)
      - Retrieval = find seed chunks → traverse graph → expand context

    Flow:
      1. retrieve()      → find seed chunks via ChromaDB
      2. _build_graph()  → build concept graph from seed chunks
      3. _expand()       → traverse graph to find related chunks
      4. augment()       → build prompt with expanded chunks
      5. generate()      → call Groq with rich connected context
    """

    def __init__(self):
        super().__init__(rag_type="graph")

        # Connect to ChromaDB
        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        # Connect to Groq LLM
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

        # Graph depth — how many hops to traverse
        self._graph_depth = int(os.getenv("GRAPH_DEPTH", "2"))

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extracts key concepts from a chunk of text.
        Uses simple tokenization — filters short/common words.
        In production, this would use NER or LLM extraction.
        """
        # Common words to ignore (stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "its"
        }

        words = text.lower().split()
        concepts = [
            word.strip(".,!?;:()[]")
            for word in words
            if len(word) > 4 and word.lower() not in stopwords
        ]

        return list(set(concepts))

    def _build_graph(
        self,
        chunks: List[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Builds a concept co-occurrence graph from chunks.

        Graph structure:
          {
            "photosynthesis": [("chlorophyll", "chunk_text"), ...],
            "chlorophyll":    [("sunlight", "chunk_text"), ...],
          }

        Two concepts are connected if they appear in the same chunk.
        The edge stores the chunk text so we can retrieve it later.
        """
        graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for chunk in chunks:
            concepts = self._extract_concepts(chunk)

            # Connect every concept to every other concept in same chunk
            for i, concept_a in enumerate(concepts):
                for concept_b in concepts[i + 1:]:
                    # Bidirectional edge — both directions
                    graph[concept_a].append((concept_b, chunk))
                    graph[concept_b].append((concept_a, chunk))

        return dict(graph)

    def _expand_graph(
        self,
        seed_chunks: List[str],
        graph: Dict[str, List[Tuple[str, str]]],
        depth: int
    ) -> List[str]:
        """
        Traverses the graph from seed chunks to find related chunks.

        Starting from concepts in seed chunks, follows edges
        to discover connected chunks up to `depth` hops away.

        Example (depth=2):
          seed: "Newton discovered laws of motion"
          hop1: finds chunks about "laws", "motion", "force"
          hop2: finds chunks about "acceleration", "mass", "velocity"
        """
        visited_concepts: Set[str] = set()
        expanded_chunks: List[str] = list(seed_chunks)

        # Extract concepts from seed chunks as starting nodes
        current_concepts = set()
        for chunk in seed_chunks:
            current_concepts.update(self._extract_concepts(chunk))

        # Traverse graph for `depth` hops
        for _ in range(depth):
            next_concepts: Set[str] = set()

            for concept in current_concepts:
                if concept in visited_concepts:
                    continue

                visited_concepts.add(concept)

                # Follow edges from this concept
                if concept in graph:
                    for neighbor_concept, edge_chunk in graph[concept]:
                        if edge_chunk not in expanded_chunks:
                            expanded_chunks.append(edge_chunk)
                        next_concepts.add(neighbor_concept)

            current_concepts = next_concepts - visited_concepts

            if not current_concepts:
                break

        return expanded_chunks

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Get seed chunks from ChromaDB.
                Build concept graph from all chunks.
                Expand via graph traversal for connected context.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Get more seed chunks than top_k for better graph coverage
            seed_results = collection.query(
                query_texts=[query.query_text],
                n_results=min(query.top_k * 2, 20)
            )

            if not seed_results["documents"] or \
               not seed_results["documents"][0]:
                return []

            seed_chunks = seed_results["documents"][0]

            # Build concept graph from seed chunks
            graph = self._build_graph(seed_chunks)

            # Expand via graph traversal
            expanded_chunks = self._expand_graph(
                seed_chunks, graph, self._graph_depth
            )

            return expanded_chunks[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[GraphRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with graph-expanded chunks.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send graph-enriched prompt to Groq and return answer.
        """
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"[GraphRAG] generate() failed: {str(e)}")