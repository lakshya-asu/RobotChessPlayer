import chess

from chess_manipulator.rl.action_space import MoveActionSpace


def test_encode_decode_round_trip_for_promotion():
    action_space = MoveActionSpace()
    move = chess.Move.from_uci("a7a8q")

    index = action_space.encode(move)
    decoded = action_space.decode(index)

    assert decoded.move == move


def test_legal_indices_include_standard_opening_move():
    action_space = MoveActionSpace()
    board = chess.Board()

    legal_indices = action_space.legal_indices(board)

    assert action_space.encode(chess.Move.from_uci("e2e4")) in legal_indices
