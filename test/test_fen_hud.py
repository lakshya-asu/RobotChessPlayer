from isaac_app.fen_hud import render_fen_board_text


def test_render_fen_board_text_renders_coordinates_and_pieces():
    rendered = render_fen_board_text("8/8/8/3p4/4P3/8/8/8 w - - 0 1")

    assert "a b c d e f g h" in rendered
    assert "5 . . . p . . . ." in rendered
    assert "4 . . . . P . . ." in rendered


def test_render_fen_board_text_defaults_empty_board_on_blank():
    rendered = render_fen_board_text("")

    assert "8 r n b q k b n r" in rendered
    assert "1 R N B Q K B N R" in rendered
