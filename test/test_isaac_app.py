from types import SimpleNamespace

import pytest

from isaac_app.channels import default_dual_robot_topic_sets
from isaac_app.chessboard import BoardGeometry, board_map_from_fen, square_color
pytest.importorskip("pxr")
from isaac_app.scene import ChessIsaacScene, PieceTransport, SceneConfig
from isaac_app.layout import default_dual_robot_placements
from isaac_app.layout import default_dual_robot_placements


def test_board_geometry_maps_all_sixty_four_squares():
    board = BoardGeometry()
    squares = board.square_map()

    assert len(squares) == 64
    assert squares["a1"][0] == board.origin_x
    assert squares["h8"][0] == board.origin_x + 7 * board.square_size_m
    assert squares["a8"][1] == board.origin_y + 7 * board.square_size_m
    assert squares["h1"][1] == board.origin_y


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


def test_default_dual_robot_placements_are_mirrored_about_board_center():
    board = BoardGeometry()
    white, black = default_dual_robot_placements(board)

    assert white.role == "white"
    assert black.role == "black"
    assert white.world_position[0] == black.world_position[0] == board.center_x
    assert white.world_position[1] < board.center_y < black.world_position[1]


def test_dual_robot_topic_sets_are_namespaced():
    topic_sets = default_dual_robot_topic_sets()

    assert topic_sets["white"].command_topic == "/white/command/joint_trajectory"
    assert topic_sets["white"].execution_result_topic == "/white/execution_result"
    assert topic_sets["black"].joint_state_topic == "/black/joint_states"
    assert topic_sets["black"].status_topic == "/black/status"


def test_dual_robot_placements_are_mirrored():
    board = BoardGeometry()
    white, black = default_dual_robot_placements(board)

    assert white.role == "white"
    assert black.role == "black"
    assert white.prim_path != black.prim_path
    assert white.world_position[0] == black.world_position[0]
    assert white.world_position[1] < board.center_y < black.world_position[1]


def test_knight_fallback_uses_capsule(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))
    calls = []

    monkeypatch.setattr(scene, "_try_create_piece_mesh", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "isaac_app.scene.VisualCapsule",
        lambda **kwargs: calls.append(("capsule", kwargs["prim_path"])),
    )
    monkeypatch.setattr(
        "isaac_app.scene.VisualCuboid",
        lambda **kwargs: calls.append(("cuboid", kwargs["prim_path"])),
    )

    scene._create_piece_visual_at_path("N", "/World/TestKnight", (0.0, 0.0, 0.0))

    assert ("capsule", "/World/TestKnight") in calls
    assert all(kind != "cuboid" for kind, _path in calls)


def test_begin_piece_transport_tracks_en_passant_capture(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))
    translations = {
        "white_pawn": (0.0, 0.0, 0.1),
        "black_pawn": (1.0, 1.0, 0.1),
    }

    monkeypatch.setattr(
        scene,
        "observed_board_map",
        lambda: {"e5": "P", "d5": "p"},
    )
    monkeypatch.setattr(
        scene,
        "_find_piece_prim_at_square",
        lambda square: {"e5": "white_pawn", "d5": "black_pawn"}.get(square),
    )
    monkeypatch.setattr(scene, "_get_prim_translation", lambda prim: translations.get(prim))
    monkeypatch.setattr(scene, "_piece_position", lambda square, symbol: (2.0, 2.0, 0.2))
    monkeypatch.setattr(scene, "_graveyard_position", lambda symbol, slot: (3.0, 3.0, 0.3))
    scene._piece_symbols_by_prim = {"white_pawn": "P", "black_pawn": "p"}

    scene.begin_piece_transport("white", "e5d6", 1.0)

    transport = scene._active_piece_transports["white"]
    assert transport.capture_square == "d5"
    assert transport.capture_prim == "black_pawn"
    assert transport.capture_symbol == "p"


def test_begin_piece_transport_fails_when_source_piece_missing(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))

    monkeypatch.setattr(scene, "observed_board_map", lambda: {})
    monkeypatch.setattr(scene, "_find_piece_prim_at_square", lambda square: None)

    assert scene.begin_piece_transport("white", "e2e4", 1.0) is False


def test_begin_piece_transport_tracks_rook_for_castling(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))

    monkeypatch.setattr(scene, "observed_board_map", lambda: {"e1": "K", "h1": "R"})
    monkeypatch.setattr(
        scene,
        "_find_piece_prim_at_square",
        lambda square: {"e1": "white_king", "h1": "white_rook"}.get(square),
    )
    monkeypatch.setattr(
        scene,
        "_get_prim_translation",
        lambda prim: {"white_king": (0.0, 0.0, 0.1), "white_rook": (1.0, 0.0, 0.1)}.get(prim),
    )
    monkeypatch.setattr(scene, "_piece_position", lambda square, symbol: (2.0, 2.0, 0.2))
    scene._piece_symbols_by_prim = {"white_king": "K", "white_rook": "R"}

    scene.begin_piece_transport("white", "e1g1", 1.0)

    transport = scene._active_piece_transports["white"]
    assert transport.rook_prim == "white_rook"
    assert transport.rook_symbol == "R"
    assert transport.rook_target_position == (2.0, 2.0, 0.2)


def test_capture_transport_uses_capturing_robot_graveyard(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))

    monkeypatch.setattr(scene, "observed_board_map", lambda: {"e4": "P", "d5": "p"})
    monkeypatch.setattr(
        scene,
        "_find_piece_prim_at_square",
        lambda square: {"e4": "white_pawn", "d5": "black_pawn"}.get(square),
    )
    monkeypatch.setattr(
        scene,
        "_get_prim_translation",
        lambda prim: {"white_pawn": (0.0, 0.0, 0.1), "black_pawn": (1.0, 0.0, 0.1)}.get(prim),
    )
    monkeypatch.setattr(scene, "_piece_position", lambda square, symbol: (2.0, 2.0, 0.2))
    monkeypatch.setattr(scene, "_graveyard_position", lambda symbol, slot, graveyard_side=None: (3.0, -1.0 if graveyard_side == "white" else 1.0, 0.3))
    scene._piece_symbols_by_prim = {"white_pawn": "P", "black_pawn": "p"}

    scene.begin_piece_transport("white", "e4d5", 1.0)

    transport = scene._active_piece_transports["white"]
    assert transport.capture_target_position == (3.0, -1.0, 0.3)


def test_finish_piece_transport_reassigns_promoted_piece(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))
    rebuilt = []

    class DummyPrim:
        def IsValid(self):
            return True

    class DummyStage:
        def GetPrimAtPath(self, prim_path):
            return DummyPrim()

        def RemovePrim(self, prim_path):
            rebuilt.append(("removed", prim_path))

    scene._world = SimpleNamespace(stage=DummyStage())
    scene._piece_pool = {"P": ["/World/ChessPieces/P_0"], "Q": ["/World/ChessPieces/Q_0"]}
    scene._piece_symbols_by_prim = {
        "/World/ChessPieces/P_0": "P",
        "/World/ChessPieces/Q_0": "Q",
    }
    monkeypatch.setattr(
        scene,
        "_create_piece_visual_at_path",
        lambda piece_symbol, prim_path, position: rebuilt.append((piece_symbol, prim_path, position)),
    )
    scene._active_piece_transports["white"] = PieceTransport(
        role="white",
        move_id="e7e8q",
        source_square="e7",
        target_square="e8",
        moving_prim="/World/ChessPieces/P_0",
        moving_symbol="P",
        final_symbol="Q",
        start_position=(0.0, 0.0, 0.1),
        target_position=(1.0, 1.0, 0.2),
        duration_sec=1.0,
    )

    scene.finish_piece_transport("white")

    assert "/World/ChessPieces/P_0" not in scene._piece_pool["P"]
    assert "/World/ChessPieces/P_0" in scene._piece_pool["Q"]
    assert scene._piece_symbols_by_prim["/World/ChessPieces/P_0"] == "Q"
    assert ("Q", "/World/ChessPieces/P_0", (1.0, 1.0, 0.2)) in rebuilt


def test_finish_piece_transport_returns_failure_when_attach_never_happens(monkeypatch):
    scene = ChessIsaacScene(SceneConfig(board_geometry=BoardGeometry()))
    restored = []

    monkeypatch.setattr(scene, "_set_prim_translation", lambda prim, pos: restored.append((prim, pos)))
    scene._active_piece_transports["white"] = PieceTransport(
        role="white",
        move_id="e2e4",
        source_square="e2",
        target_square="e4",
        moving_prim="/World/ChessPieces/P_0",
        moving_symbol="P",
        final_symbol="P",
        start_position=(0.0, 0.0, 0.1),
        target_position=(1.0, 1.0, 0.2),
        duration_sec=1.0,
        failure_reason="failed to attach piece for e2e4",
    )

    result = scene.finish_piece_transport("white")

    assert result.success is False
    assert "failed to attach piece" in result.message
    assert ("/World/ChessPieces/P_0", (0.0, 0.0, 0.1)) in restored
