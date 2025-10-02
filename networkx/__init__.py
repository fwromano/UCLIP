"""Minimal networkx stub to satisfy optional dependencies."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, Iterator, MutableMapping, Tuple, TypeVar


__all__ = [
    "DiGraph",
    "bfs_predecessors",
    "spring_layout",
    "draw_networkx_nodes",
    "draw_networkx_edges",
    "draw_networkx_labels",
]

__version__ = "0.0-stub"

_NodeKey = TypeVar("_NodeKey")


class _NodeView(MutableMapping[_NodeKey, dict]):
    def __init__(self, storage: Dict[_NodeKey, dict]):
        self._storage = storage

    def __getitem__(self, key: _NodeKey) -> dict:
        return self._storage.setdefault(key, {})

    def __setitem__(self, key: _NodeKey, value: dict) -> None:
        self._storage[key] = value

    def __delitem__(self, key: _NodeKey) -> None:
        del self._storage[key]

    def __iter__(self) -> Iterator[_NodeKey]:
        return iter(self._storage)

    def __len__(self) -> int:
        return len(self._storage)


class DiGraph:
    def __init__(self) -> None:
        self._succ: Dict[_NodeKey, set[_NodeKey]] = {}
        self._pred: Dict[_NodeKey, set[_NodeKey]] = {}
        self._node_data: Dict[_NodeKey, dict] = {}

    def add_node(self, node: _NodeKey, **attrs) -> None:
        self._succ.setdefault(node, set())
        self._pred.setdefault(node, set())
        data = self._node_data.setdefault(node, {})
        data.update(attrs)

    def add_nodes_from(self, nodes: Iterable[_NodeKey]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: _NodeKey, v: _NodeKey, **attrs) -> None:
        self.add_node(u)
        self.add_node(v)
        self._succ[u].add(v)
        self._pred[v].add(u)
        if attrs:
            self._node_data[v].update(attrs)

    @property
    def nodes(self) -> _NodeView[_NodeKey]:
        return _NodeView(self._node_data)

    def predecessors(self, node: _NodeKey) -> Iterator[_NodeKey]:
        return iter(self._pred.get(node, ()))

    def successors(self, node: _NodeKey) -> Iterator[_NodeKey]:
        return iter(self._succ.get(node, ()))

    def edges(self) -> Iterator[Tuple[_NodeKey, _NodeKey]]:
        for source, targets in self._succ.items():
            for target in targets:
                yield (source, target)


def bfs_predecessors(graph: DiGraph, source: _NodeKey):
    visited = {source}
    queue: deque[tuple[_NodeKey, _NodeKey | None]] = deque([(source, None)])
    while queue:
        node, parent = queue.popleft()
        for succ in graph._succ.get(node, ()):  # type: ignore[attr-defined]
            if succ in visited:
                continue
            visited.add(succ)
            queue.append((succ, node))
            yield succ, node  # type: ignore[misc]


def spring_layout(graph: DiGraph, k: float | None = None, iterations: int | None = None):
    nodes = list(graph.nodes)
    total = len(nodes)
    if total == 0:
        return {}
    radius = 1.0
    return {
        node: (
            radius * math.cos((2 * math.pi * idx) / total),
            radius * math.sin((2 * math.pi * idx) / total),
        )
        for idx, node in enumerate(nodes)
    }


def draw_networkx_nodes(*_args, **_kwargs) -> None:  # pragma: no cover - visual helper
    return None


def draw_networkx_edges(*_args, **_kwargs) -> None:  # pragma: no cover - visual helper
    return None


def draw_networkx_labels(*_args, **_kwargs) -> None:  # pragma: no cover - visual helper
    return None


def __getattr__(name: str):  # pragma: no cover - compatibility shim
    raise AttributeError(f"networkx stub does not provide attribute {name!r}")

