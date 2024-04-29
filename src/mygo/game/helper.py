from mygo.game.types import Game, Move, Player


def move_to_str(player: Player, move: Move, i: int = 0) -> str:
    """Return human-friendly representation of move."""
    idx = "" if i == 0 else f"({i:3})"
    return f"{str(player)}{idx}: {move}"


def game_to_str(game: Game, i: int = 0) -> str:
    """Return human-friendly representation of game."""
    return f"{game}\n\n{move_to_str(-game.next_player, game.move, i)}"
