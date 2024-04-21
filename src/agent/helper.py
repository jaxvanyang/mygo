from go.types import Color, Move


def print_move(color: Color, move: Move, i: int = 0) -> None:
    print(f"{'black' if color == Color.black else 'white'}({i:3}): {move}")
