import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from pysgf import SGFNode

from mygo import __version__
from mygo.agent import MCTSBot, MLBot, RandomBot, TreeSearchBot
from mygo.agent.base import Agent
from mygo.encoder.oneplane import OnePlaneEncoder
from mygo.game.types import Game, Move, Player, Point
from mygo.model import SmallModel, TinyModel


class CommandEffect(Enum):
    """The effect to the game of applying a command."""

    no_effect = 0
    next_round = 1
    end_game = 2


class Command:
    "ASCII mode command."

    types = ("help", "exit", "quit", "pass", "resign", "save", "move")
    help_msg = """Commands:
    help\t\tDisplay this help menu
    exit\t\tExit MyGo
    quit\t\tExit MyGo
    pass\t\tPass on your move
    resign\t\tResign the current game
    save <file>\t\tSave the current game
    <move>\t\tA move of the format <letter><number>
    """

    def __init__(self, command_type: str, arg: Any = None) -> None:
        self.type = command_type
        self.arg = arg

    @staticmethod
    def _parse_point(gtp_coords: str) -> Point | None:
        """Create a point from GTP coordinates.

        Return None if cannot parse.
        """

        gtp_coords = gtp_coords.upper()
        if match := re.match(f"([{Move._COLS}])([\\d]+)", gtp_coords):
            return Point(int(match[2]), Move._COLS.index(match[1]) + 1)
        else:
            return None

    @classmethod
    def parse_command(cls, command_str: str):
        """Parse a command from the input string.

        Return None if it's invalid.
        """
        command, *args = command_str.split(" ")

        for c in cls.types[:-1]:
            if c.startswith(command):
                command = c
                break

        if args:
            if len(args) == 1 and command == "save":
                return cls(command, Path(args[0]))
            return None

        if command in cls.types[:5]:
            return cls(command)

        if point := cls._parse_point(command):
            return cls("move", point)

        return None

    def apply(self, game: Game) -> CommandEffect:
        """Apply this command to game.

        Return the effect to the game of applying this command.
        """

        match self.type:
            case "help":
                print(self.help_msg)
                return CommandEffect.no_effect
            case "exit" | "quit":
                print("Thanks for playing MyGo!\n")
                return CommandEffect.end_game
            case "pass":
                game.apply_move(Move.pass_())
                return CommandEffect.next_round
            case "resign":
                game.apply_move(Move.resign())
                return CommandEffect.end_game
            case "save":
                raise NotImplementedError()
            case "move":
                game.apply_move(Move.play(self.arg))
                return CommandEffect.next_round

        raise RuntimeError(f"Unexpected command: {self.type}")


class MyGo:
    """MyGo application."""

    def __init__(
        self,
        size: int = 19,
        color: str = "black",
        handicap: int = 0,
        komi: float = 5.5,
        mode: str = "ascii",
        outfile: Path | None = None,
        bot: str = "random",
        bot_args: list[str] = [],
        weights: Path | None = None,
    ) -> None:
        assert handicap >= 0
        assert komi >= 0.0

        self.logger = logging.getLogger("mygo")
        self.size = size
        self.human_player = Player.black if color == "black" else Player.white
        self.computer_player = -self.human_player
        self.handicap = handicap
        self.komi = komi
        self.mode = mode
        self.outfile = outfile
        self.bot = self._load_bot(bot, bot_args, weights)

    def _load_bot(
        self, bot_name: str, bot_args: list[str], weights: Path | None
    ) -> Agent:
        match bot_name:
            case "random":
                if bot_args:
                    self.logger.warning(f"random bot doesn't use {bot_args=}")
                if weights:
                    self.logger.warning(f"random bot doesn't use {weights=}")
                return RandomBot()
            case "minimax":
                if weights:
                    self.logger.warning(f"minimax bot doesn't use {weights=}")
                match bot_args:
                    case []:
                        return TreeSearchBot()
                    case [depth]:
                        return TreeSearchBot(int(depth))
                    case [depth, *args]:
                        self.logger.warning(f"minimax bot doesn't use extra {args=}")
                        return TreeSearchBot(int(depth))
                    case _:
                        raise ValueError(f"{bot_args=} is not a list")
            case "mcts":
                if weights:
                    self.logger.warning(f"mcts bot doesn't use {weights=}")
                match bot_args:
                    case []:
                        return MCTSBot()
                    case [rounds]:
                        return MCTSBot(int(rounds))
                    case [rounds, temp]:
                        return MCTSBot(int(rounds), float(temp))
                    case [rounds, temp, resign_rate]:
                        return MCTSBot(int(rounds), float(temp), float(resign_rate))
                    case [rounds, temp, resign_rate, *args]:
                        self.logger.warning(f"mcts bot doesn't use extra {args=}")
                        return MCTSBot(int(rounds), float(temp), float(resign_rate))
                    case _:
                        raise ValueError(f"{bot_args=} is not a list")
            case "tiny":
                if bot_args:
                    self.logger.warning(f"tiny bot doesn't use {bot_args=}")
                model = TinyModel(self.size)
                if weights:
                    model.load_state_dict(torch.load(weights))
                else:
                    raise NotImplementedError("auto download pre-trained weights")
                return MLBot(model, OnePlaneEncoder(self.size))
            case "small":
                if bot_args:
                    self.logger.warning(f"small bot doesn't use {bot_args=}")
                model = SmallModel(self.size)
                if weights:
                    model.load_state_dict(torch.load(weights))
                else:
                    raise NotImplementedError("auto download pre-trained weights")
                return MLBot(model, OnePlaneEncoder(self.size))
            case _:
                raise ValueError(f"bot not supported: {bot_name}")

    def run_ascii(self) -> SGFNode:
        """Run ASCII mode game.

        Return the root SGFNode of this game.
        """

        sgf_root = SGFNode()
        game = Game.new_game(self.size)
        move_number = 0
        black_captures, white_captures = 0, 0

        if self.handicap and game.next_player == self.computer_player:
            print("MyGo is placing free handicap...\n")
            for _ in range(self.handicap):
                game.next_player = self.computer_player
                move = self.bot.select_move(game)
                assert move.is_play
                game.apply_move(move)

            move_number += self.handicap

        print(
            f"MyGo {__version__}\n",
            "Beginning ASCII mode game.\n",
            f"Board Size:\t\t{self.size}",
            f"Handicap:\t\t{self.handicap}",
            f"Komi:\t\t\t{self.komi}",
            f"Move Number:\t\t{move_number}",
            f"To Move:\t\t{game.next_player}",
            f"Computer Player:\t{self.computer_player}",
            sep="\n",
            end="\n\n",
        )

        if self.handicap and game.next_player == self.human_player:
            while move_number < self.handicap:
                game.next_player = self.human_player
                print(f"{game}\n")

                try:
                    input_str = input(f"{game.next_player}({move_number + 1}): ")
                except EOFError:
                    input_str = "exit"

                print()

                command = Command.parse_command(input_str)

                if command is None:
                    print(f"Invalid command: {input_str}\n")
                    continue

                if command.type == "pass" or command.type == "resign":
                    print(f"You cannot {command.type} on a handicap round!\n")
                    continue

                if command.type == "move" and not game.board.is_placeable(command.arg):
                    print(f"Invalid move: {input_str}\n")
                    continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    move_number += 1
                elif effect == CommandEffect.end_game:
                    return sgf_root

        while True:
            print(
                f"Black (X) has captured {black_captures} pieces",
                f"White (O) has captured {white_captures} pieces",
                sep="\n",
                end="\n\n",
            )
            print(f"{game}\n")

            if game.next_player == self.human_player:
                try:
                    input_str = input(f"{game.next_player}({move_number + 1}): ")
                except EOFError:
                    input_str = "exit"

                print()

                command = Command.parse_command(input_str)

                if command is None:
                    print(f"Invalid command: {input_str}\n")
                    continue

                if command.type == "move":
                    if game.board.is_placeable(command.arg):
                        move = Move.play(command.arg)
                        if self.computer_player == Player.black:
                            black_captures += game.apply_move(move)
                        else:
                            white_captures += game.apply_move(move)
                        move_number += 1
                    else:
                        print(f"Invalid move: {input_str}\n")
                    continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    move_number += 1
                elif effect == CommandEffect.end_game:
                    return sgf_root
            else:
                print("MyGo is thinking...\n")
                move = self.bot.select_move(game)
                if self.computer_player == Player.black:
                    black_captures += game.apply_move(move)
                else:
                    white_captures += game.apply_move(move)
                move_number += 1
                print(f"{self.computer_player}({move_number}): {move}\n")

                if game.is_over:
                    return sgf_root

    def run_gtp(self) -> SGFNode:
        raise NotImplementedError()

    def start(self) -> int:
        """Start playing."""

        match self.mode:
            case "ascii":
                sgf_root = self.run_ascii()
            case "gtp":
                sgf_root = self.run_ascii()
            case _:
                raise ValueError(f"mode not supported: {self.mode}")

        # TODO: print game result

        if self.outfile:
            with open(self.outfile, "w", encoding="utf-8") as f:
                f.write(sgf_root.sgf())

        return 0
