"""Vector search tool for querying business rules and documentation."""

from typing import Any

import chromadb

from config import settings
from tools.base import BaseTool, ToolResult


class VectorSearchTool(BaseTool):
    name = "vector_search"
    description = (
        "Search business documentation, rules, and KPI definitions using natural language queries. "
        "Returns the most relevant document chunks."
    )

    def __init__(self, persist_dir: str | None = None) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_collection("business_docs")
        return self._collection

    async def execute(self, query: str, n_results: int = 3, **kwargs: Any) -> ToolResult:
        try:
            collection = self._get_collection()
            results = collection.query(query_texts=[query], n_results=n_results)

            documents = []
            ids_data = results.get("ids") or []
            ids = ids_data[0] if ids_data else []
            docs_data = results.get("documents") or []
            docs_list = docs_data[0] if docs_data else []
            meta_data = results.get("metadatas") or []
            metadata_list = meta_data[0] if meta_data else []
            dist_data = results.get("distances") or []
            distances = dist_data[0] if dist_data else []

            for i in range(len(ids)):
                documents.append({
                    "id": ids[i],
                    "text": docs_list[i],
                    "metadata": metadata_list[i] if i < len(metadata_list) else {},
                    "distance": distances[i] if i < len(distances) else None,
                })

            return ToolResult(success=True, data={"query": query, "results": documents})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
