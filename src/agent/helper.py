from go.types import Color, Move


def print_move(color: Color, move: Move, i: int = 0) -> None:
    idx = "" if i == 0 else f"({i:3})"
    print(f"{str(color)}{idx}: {move}")
