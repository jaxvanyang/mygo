#!/usr/bin/env python3

import shlex
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from subprocess import PIPE, Popen

from mygo import pysgf
from mygo.game.types import Game, Move, Player
from mygo.pysgf.parser import SGFNode


class Bot:
    def __init__(self, command: str, player: Player) -> None:
        self.player = player
        self.engine = Popen(
            shlex.split(command),
            stdin=PIPE,
            stdout=PIPE,
            encoding="utf-8",
        )

    def send_command(self, command: str) -> None:
        self.engine.stdin.write(f"{command}\n")  # pytype: disable=attribute-error
        self.engine.stdin.flush()  # pytype: disable=attribute-error

    def get_response(self) -> str:
        response = ""
        while (
            line := self.engine.stdout.readline()  # pytype: disable=attribute-error
        ) != "\n":
            response += line
        return response

    def commnunicate(self, command: str) -> str:
        self.send_command(command)
        return self.get_response()


def bvb(game: Game, sgf_root: SGFNode, black_bot: Bot, white_bot: Bot) -> None:
    sgf_node = sgf_root
    bots = (black_bot, white_bot)

    i = 0
    while not game.is_over:
        assert game.next_player == bots[i].player

        j = i ^ 1

        command = f"genmove {bots[i].player}"
        print(f"{bots[i].player}> {command}")
        response = bots[i].commnunicate(command)
        print(response, end="")
        assert response.startswith("=")

        vertex = response.split()[-1]
        move = Move.from_gtp(vertex)
        game.apply_move(move)

        if move.is_resign:
            sgf_root.set_property("RE", f"{bots[j].player.sgf}+Resign")
            return

        sgf_node = sgf_node.play(pysgf.Move.from_gtp(vertex, player=bots[i].player.sgf))

        command = f"play {bots[i].player} {vertex}"
        print(f"{bots[j].player}> {command}")
        response = bots[j].commnunicate(command)
        print(response, end="")
        assert response.startswith("=")

        i = j


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(
        "bvb", description="Bot vs. bot. Create game between GTP engines."
    )
    parser.add_argument(
        "--size",
        "--boardsize",
        default=19,
        type=int,
        choices=tuple(range(1, 20)),
        help="Set the board size to use (1-19). Default is 19.",
        metavar="num",
    )
    parser.add_argument(
        "--black",
        default="mygo --mode gtp",
        help='Set the GTP engine as the black player. Default is "mygo --mode gtp".',
        metavar="command",
    )
    parser.add_argument(
        "--white",
        default="mygo --mode gtp",
        help='Set the GTP engine as the white player. Default is "mygo --mode gtp".',
        metavar="command",
    )
    parser.add_argument(
        "--komi",
        default=5.5,
        type=float,
        help=(
            "Set the komi (points given to white player to compensate advantage of the "
            "first move, usually 5.5 or 0.5). Default is 5.5."
        ),
        metavar="num",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="Save the played game to file in SGF format.",
        metavar="file",
    )

    parsed_args = parser.parse_args(args)
    black_bot = Bot(parsed_args.black, Player.black)
    white_bot = Bot(parsed_args.white, Player.white)

    command = f"boardsize {parsed_args.size}"
    print(f"black> {command}")
    print(black_bot.commnunicate(command), end="")
    print(f"white> {command}")
    print(white_bot.commnunicate(command), end="")

    command = f"komi {parsed_args.komi}"
    print(f"black> {command}")
    print(black_bot.commnunicate(command), end="")
    print(f"white> {command}")
    print(white_bot.commnunicate(command), end="")

    game = Game.new_game(parsed_args.size, komi=parsed_args.komi)
    sgf_root = SGFNode(
        properties={
            "GM": 1,
            "FF": 4,
            "SZ": parsed_args.size,
            "DT": date.today().strftime("%Y-%m-%d"),
            "KM": parsed_args.komi,
            "CA": "UTF-8",
            "PB": parsed_args.black,
            "PW": parsed_args.white,
        }
    )
    bvb(game, sgf_root, black_bot, white_bot)

    result = game.result
    print(f"Result: {result}")

    if parsed_args.outfile:
        sgf_root.set_property("RE", result)
        with open(parsed_args.outfile, "w", encoding="utf-8") as f:
            f.write(sgf_root.sgf())

    return 0


if __name__ == "__main__":
    sys.exit(main())
