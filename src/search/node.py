"""MCTS node."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class Node:
    board: chess.Board
    prior: float = 0.0
    move: chess.Move | None = None
    parent: "Node | None" = None
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visit_count == 0 else self.total_value / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    def backup(self, value: float) -> None:
        node: Node | None = self
        sign = 1.0
        while node is not None:
            node.visit_count += 1
            node.total_value += sign * value
            sign *= -1.0
            node = node.parent
