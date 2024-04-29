from mygo.game.types import Color, Game, Move


def move_to_str(color: Color, move: Move, i: int = 0) -> str:
    """Return human-friendly representation of move."""
    idx = "" if i == 0 else f"({i:3})"
    return f"{str(color)}{idx}: {move}"


def game_to_str(game: Game, i: int = 0) -> str:
    """Return human-friendly representation of game."""
    return f"{game}\n\n{move_to_str(game.next_color.opposite, game.move, i)}"
