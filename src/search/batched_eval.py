"""Model evaluation helpers for MCTS."""

from __future__ import annotations

import chess
import torch
import torch.nn.functional as F

from chess_env.action_space import legal_moves_to_action_ids
from chess_env.board_encoding import BoardEncodingConfig, encode_board_planes, encode_board_sequence
from chess_env.legal_mask import make_legal_mask, mask_illegal_logits


class ModelEvaluator:
    def __init__(self, model, device: torch.device, model_type: str, shuffled: bool = False):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.enc_cfg = BoardEncodingConfig(shuffled_order=shuffled)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[dict[int, float], float]:
        mask = make_legal_mask(board, self.device)
        if self.model_type.startswith("cnn"):
            planes = torch.from_numpy(encode_board_planes(board)).unsqueeze(0).to(self.device)
            logits, value = self.model(planes)
        else:
            enc = encode_board_sequence(board, self.enc_cfg)
            tokens = torch.from_numpy(enc["tokens"]).unsqueeze(0).to(self.device)
            rows = torch.from_numpy(enc["rows"]).unsqueeze(0).to(self.device)
            cols = torch.from_numpy(enc["cols"]).unsqueeze(0).to(self.device)
            logits, value = self.model(tokens, rows, cols)
        probs = F.softmax(mask_illegal_logits(logits, mask.unsqueeze(0)), dim=-1).squeeze(0)
        priors = {aid: float(probs[aid].item()) for aid in legal_moves_to_action_ids(board)}
        return priors, float(value.squeeze().item())
