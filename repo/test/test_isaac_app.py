from isaac_app.chessboard import BoardGeometry, board_map_from_fen, square_color


def test_board_geometry_maps_all_sixty_four_squares():
    board = BoardGeometry()
    squares = board.square_map()

    assert len(squares) == 64
    assert squares["a1"][0] == board.origin_x
    assert squares["h8"][0] == board.origin_x + 7 * board.square_size_m
    assert squares["a8"][1] == board.origin_y


def test_square_color_alternates():
    assert square_color("a1").tolist() != square_color("b1").tolist()
    assert square_color("a1").tolist() == square_color("c1").tolist()


def test_fen_parser_maps_starting_position():
    board_map = board_map_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    assert len(board_map) == 32
    assert board_map["a8"] == "r"
    assert board_map["e1"] == "K"
    assert "e4" not in board_map


def test_fen_parser_maps_moved_pawn():
    board_map = board_map_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")

    assert board_map["e4"] == "P"
    assert "e2" not in board_map
