import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from mygo import __version__
from mygo.cli.app import MyGo


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(
        "mygo",
        description="My BetaGo implementation in PyTorch!",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version and exit.",
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
        "--color",
        default="black",
        type=str.lower,
        choices=("black", "white"),
        help=(
            "Choose your color (black or white). Black plays first, white gets the "
            "komi compensation. Default is black."
        ),
        metavar="color",
    )
    parser.add_argument(
        "--handicap",
        default=0,
        type=int,
        help="Set the number of handicap stones. Default is 0.",
        metavar="num",
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
        "--mode",
        default="ascii",
        type=str.lower,
        choices=("ascii", "gtp"),
        help="Set the playing mode (ascii or gtp). Default is ASCII.",
        metavar="mode",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="Save the played game to file in SGF format.",
        metavar="file",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set the logging level. Default is WARNING.",
        metavar="level",
    )
    parser.add_argument(
        "-b",
        "--bot",
        default="random",
        type=str.lower,
        choices=("random", "minimax", "mcts", "tiny", "small"),
        help=(
            "Set the bot to play with. Avaliable bots: random, minimax, mcts, tiny, "
            "small. Default is random."
        ),
        metavar="bot",
    )
    # TODO: add description of each bot in a seperate section
    parser.add_argument(
        "--bot-args",
        default=[],
        type=lambda x: x.split(","),
        help=(
            "Set optional arguments of the bot. This option requires a compatible bot. "
            "The input should be in CSV (comma-seperated values) format."
        ),
        metavar="arg1,...",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        help="Load model weights from file. This option requires a compatible bot.",
        metavar="file",
    )

    parsed_args = parser.parse_args(args)
    if parsed_args.bot_args:
        raise NotImplementedError('option "--bot-args" is not implemented')

    logger = logging.getLogger("mygo")
    logger.setLevel(parsed_args.log_level)

    kwargs = vars(parsed_args)
    del kwargs["log_level"]

    app = MyGo(**kwargs)
    return app.start()


if __name__ == "__main__":
    sys.exit(main())
