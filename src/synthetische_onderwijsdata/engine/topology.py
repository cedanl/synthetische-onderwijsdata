"""Topological traversal of the relational schema DAG (Kahn's algorithm)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set


class SchemaGraph:
    """
    Directed acyclic graph where edges point from parent (dimension) to
    child (fact) tables.  Topological sort guarantees parents are generated
    before children, satisfying referential integrity constraints.
    """

    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._in_degree: Dict[str, int] = defaultdict(int)
        self._tables: Set[str] = set()

    def add_table(self, name: str) -> None:
        self._tables.add(name)
        self._in_degree.setdefault(name, 0)

    def add_dependency(self, parent: str, child: str) -> None:
        """Register that *parent* must be generated before *child*."""
        for t in (parent, child):
            self._tables.add(t)
            self._in_degree.setdefault(t, 0)
        if child not in self._children[parent]:
            self._children[parent].add(child)
            self._in_degree[child] += 1

    def topological_order(self) -> List[str]:
        """Return tables in generation order.  Raises if the graph has a cycle."""
        in_degree = dict(self._in_degree)
        queue: deque[str] = deque(
            sorted(t for t in self._tables if in_degree[t] == 0)
        )
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in sorted(self._children.get(node, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._tables):
            cycle_nodes = self._tables - set(order)
            raise ValueError(
                f"Schema contains a cycle involving tables: {cycle_nodes}"
            )
        return order
