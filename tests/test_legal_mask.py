import chess
import torch

from chess_env.action_space import ACTION_SIZE, move_to_action_id
from chess_env.legal_mask import make_legal_mask, mask_illegal_logits


def test_starting_position_mask_count():
    mask = make_legal_mask(chess.Board())
    assert mask.shape == (ACTION_SIZE,)
    assert int(mask.sum().item()) == 20


def test_mate_and_stalemate_masks_empty():
    mate = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert int(make_legal_mask(mate).sum().item()) == 0
    assert int(make_legal_mask(stale).sum().item()) == 0


def test_illegal_logits_have_no_gradient_when_masked():
    board = chess.Board()
    target = torch.tensor([move_to_action_id(board, chess.Move.from_uci("e2e4"))])
    mask = make_legal_mask(board).unsqueeze(0)
    logits = torch.randn(1, ACTION_SIZE, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(mask_illegal_logits(logits, mask), target)
    loss.backward()
    illegal_grad = logits.grad[0, ~mask.squeeze(0)]
    assert torch.all(illegal_grad == 0)
