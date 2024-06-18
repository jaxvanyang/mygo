import logging
import readline  # noqa: F401
from datetime import date
from pathlib import Path

# TODO: lazy import
import torch

from mygo import __version__, pysgf
from mygo.agent import MCTSBot, MLBot, RandomBot, TreeSearchBot
from mygo.agent.base import Agent
from mygo.cli.command import ASCIICommand, CommandEffect, GTPCommand
from mygo.encoder.oneplane import OnePlaneEncoder
from mygo.game.basic import Player, Point
from mygo.game.game import Game
from mygo.game.move import PassMove, PlayMove
from mygo.model import SmallModel, TinyModel
from mygo.pysgf import SGFNode


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

        move_number = 0
        black_captures, white_captures = 0, 0
        game = Game.new(self.size, komi=self.komi)
        sgf_root = SGFNode(
            properties={
                "GM": 1,
                "FF": 4,
                "SZ": self.size,
                "DT": date.today().strftime("%Y-%m-%d"),
                "KM": self.komi,
                "CA": "UTF-8",
                "HA": self.handicap,
                f"P{self.human_player.sgf}": "Human Player",
                f"P{self.computer_player.sgf}": self.bot.name,
            }
        )
        sgf_node = sgf_root

        if self.handicap and game.next_player == self.computer_player:
            print("MyGo is placing free handicap...\n")
            for _ in range(self.handicap):
                game.next_player = self.computer_player
                move = self.bot.select_move(game)

                assert move.is_play
                game.apply_move(move)
                sgf_root.add_property("AB", move.sgf(self.size))

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
                    command_str = input(f"{game.next_player}({move_number + 1}): ")
                except EOFError:
                    command_str = "exit"

                print()

                command = ASCIICommand.parse(command_str)

                if command is None:
                    print(f"Invalid command: {command_str}\n")
                    continue

                match command.name:
                    case "pass" | "resign":
                        print(f"You cannot {command.name} on a handicap round!\n")
                        continue
                    case "move" if not game.last_board.is_placeable(command.arg):
                        print(f"Invalid move: {command_str}\n")
                        continue
                    case "save":
                        try:
                            with open(command.arg, "w", encoding="utf-8") as f:
                                f.write(sgf_root.sgf())
                            print(f"Current game is saved to {command.arg}.\n")
                        except OSError:
                            print(f"Failed to save current game to {command.arg}.\n")
                        continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    move_number += 1
                    sgf_root.add_property("AB", command.arg.sgf(self.size))
                elif effect == CommandEffect.end_game:
                    return sgf_root

        while not game.is_over:
            print(
                f"Black (X) has captured {black_captures} pieces",
                f"White (O) has captured {white_captures} pieces",
                sep="\n",
                end="\n\n",
            )
            print(f"{game}\n")

            if game.next_player == self.human_player:
                try:
                    command_str = input(f"{game.next_player}({move_number + 1}): ")
                except EOFError:
                    command_str = "exit"

                print()

                command = ASCIICommand.parse(command_str)

                if command is None:
                    print(f"Invalid command: {command_str}\n")
                    continue

                match command.name:
                    case "move":
                        if game.last_board.is_placeable(command.arg):
                            move = PlayMove(game.next_player, command.arg)
                            if self.computer_player == Player.black:
                                black_captures += game.apply_move(move)
                            else:
                                white_captures += game.apply_move(move)
                            move_number += 1
                            sgf_node = sgf_node.play(
                                pysgf.Move.from_sgf(
                                    command.arg.sgf(self.size),
                                    (self.size, self.size),
                                    self.human_player.sgf,
                                )
                            )
                        else:
                            print(f"Invalid move: {command_str}\n")
                        continue
                    case "save":
                        try:
                            with open(command.arg, "w", encoding="utf-8") as f:
                                f.write(sgf_root.sgf())
                            print(f"Current game is saved to {command.arg}.\n")
                        except OSError:
                            print(f"Failed to save current game to {command.arg}.\n")
                        continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    # this is a pass move
                    sgf_node = sgf_node.play(pysgf.Move(player=self.human_player.sgf))
                    move_number += 1
                elif effect == CommandEffect.end_game:
                    if command.name == "resign":
                        sgf_root.add_property("RE", f"{game.next_player.sgf}+Resign")
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

                if move.is_resign:
                    sgf_root.add_property("RE", f"{game.next_player.sgf}+Resign")
                else:
                    sgf_node = sgf_node.play(
                        pysgf.Move.from_sgf(
                            move.sgf(self.size),
                            (self.size, self.size),
                            self.computer_player.sgf,
                        )
                    )

        sgf_root.set_property("RE", game.result)
        return sgf_root

    def run_gtp(self) -> SGFNode:
        """Run GTP mode game.

        Return the root SGFNode of this game.
        """

        game = Game.new(self.size, komi=self.komi)
        sgf_root = SGFNode(
            properties={
                "GM": 1,
                "FF": 4,
                "SZ": self.size,
                "DT": date.today().strftime("%Y-%m-%d"),
                "KM": self.komi,
                "CA": "UTF-8",
                "HA": self.handicap,
                f"P{self.human_player.sgf}": "Human Player",
                f"P{self.computer_player.sgf}": self.bot.name,
            }
        )
        sgf_node = sgf_root

        while True:
            try:
                command_str = input().strip()
            except EOFError:
                return sgf_root
            if not command_str:
                continue
            command = GTPCommand.parse(command_str)

            match command.name:
                case "clear_board":
                    game.reset(game.board_size)
                    sgf_root.children = []
                    sgf_node = sgf_root
                    command.print_output()
                    continue
                case "komi":
                    try:
                        komi = float(command.args[0])
                    except (TypeError, IndexError, ValueError):
                        command.print_output("komi not a float", success=False)
                    else:
                        sgf_root.set_property("KM", komi)
                        command.print_output()
                    continue
                case "play":
                    try:
                        color, vertex, *_ = command.args
                        match color.lower():
                            case "b" | "black":
                                player = Player.black
                            case "w" | "white":
                                player = Player.white
                            case _:
                                raise ValueError(f"Invalid color: {color}")
                    except (TypeError, ValueError):
                        command.print_output(
                            "invalid color or coordinates", success=False
                        )
                        continue

                    if vertex.lower() == "pass":
                        game.next_player = player
                        command.print_output()
                        game.apply_move(PassMove(player))
                        sgf_node = sgf_node.play(pysgf.Move(player=player.sgf))
                        continue

                    try:
                        point = Point.from_gtp(vertex)
                    except ValueError:
                        command.print_output("invalid coordinates", success=False)
                        continue

                    old_player = game.next_player
                    game.next_player = player
                    if not game.is_valid_move(move := PlayMove(player, point)):
                        game.next_player = old_player
                        command.print_output("illegal move", success=False)
                        continue

                    game.apply_move(move)
                    sgf_node = sgf_node.play(
                        pysgf.Move.from_gtp(vertex.upper(), player=player.sgf)
                    )
                    command.print_output()
                    continue
                case "genmove":
                    try:
                        color = command.args[0].lower()
                    except (TypeError, IndexError):
                        command.print_output("invalid color", success=False)
                        continue

                    match color:
                        case "b" | "black":
                            player = Player.black
                        case "w" | "white":
                            player = Player.white
                        case _:
                            command.print_output("invalid color", success=False)
                            continue

                    game.next_player = player
                    move = self.bot.select_move(game)
                    game.apply_move(move)
                    command.print_output(str(move))
                    if move.is_resign:
                        return sgf_root

                    sgf_node = sgf_node.play(move.to_pysgf())
                    continue

            # other commands are handled manually
            if command.apply(game) == CommandEffect.end_game:
                return sgf_root

    def start(self) -> int:
        """Start playing."""

        match self.mode:
            case "ascii":
                sgf_root = self.run_ascii()
            case "gtp":
                sgf_root = self.run_gtp()
            case _:
                raise ValueError(f"mode not supported: {self.mode}")

        # TODO: print game result

        if self.outfile:
            with open(self.outfile, "w", encoding="utf-8") as f:
                f.write(sgf_root.sgf())

        return 0
