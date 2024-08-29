import logging
import readline  # noqa: F401
from datetime import date
from pathlib import Path

# TODO: lazy import
import torch

from mygo import __version__
from mygo.agent import MCTSBot, MLBot, RandomBot, TreeSearchBot
from mygo.agent.base import Agent
from mygo.cli.command import ASCIICommand, CommandEffect, GTPCommand
from mygo.encoder.oneplane import OnePlaneEncoder
from mygo.game.basic import Player
from mygo.game.game import Game
from mygo.game.move import PlayMove, from_gtp_move
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
        self.handicap = 0 if mode == "ascii" else handicap  # not supported in GTP mode
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

    def _save_sgf(self, sgf_root: SGFNode, path: Path) -> None:
        """Add game information to the sgf_root and save it to path."""
        sgf_root.set_property("DT", date.today().strftime("%Y-%m-%d"))
        sgf_root.set_property(f"P{self.human_player.sgf}", "Human Player")
        sgf_root.set_property(f"P{self.computer_player.sgf}", self.bot.name)
        if self.handicap:
            sgf_root.set_property("HA", self.handicap)

        with open(path, "w", encoding="utf-8") as f:
            f.write(sgf_root.sgf())

    def run_ascii(self) -> Game:
        """Run ASCII mode game.

        Return:
            A generated Game instance.
        """

        move_number = 0
        black_captures, white_captures = 0, 0
        game = Game.new(self.size, komi=self.komi)

        if self.handicap and game.next_player == self.computer_player:
            print("MyGo is placing free handicap...\n")
            for _ in range(self.handicap):
                move = self.bot.select_move(game, self.computer_player)

                assert move.is_play
                game.last_board.apply_move(move)

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
                            self._save_sgf(game.to_pysgf(), command.arg)
                            print(f"Current game is saved to {command.arg}.\n")
                        except OSError:
                            print(f"Failed to save current game to {command.arg}.\n")
                        continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    move_number += 1
                elif effect == CommandEffect.end_game:
                    return game

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
                        else:
                            print(f"Invalid move: {command_str}\n")
                        continue
                    case "save":
                        try:
                            self._save_sgf(game.to_pysgf(), command.arg)
                            print(f"Current game is saved to {command.arg}.\n")
                        except OSError:
                            print(f"Failed to save current game to {command.arg}.\n")
                        continue

                effect = command.apply(game)
                if effect == CommandEffect.next_round:
                    # this is a pass move
                    move_number += 1
                elif effect == CommandEffect.end_game:
                    return game
            else:
                print("MyGo is thinking...\n")
                move = self.bot.select_move(game, self.computer_player)
                if self.computer_player == Player.black:
                    black_captures += game.apply_move(move)
                else:
                    white_captures += game.apply_move(move)

                move_number += 1
                print(f"{self.computer_player}({move_number}): {move}\n")

        return game

    def run_gtp(self) -> Game:
        """Run GTP mode game.

        Return:
            A generated Game instance.
        """

        game = Game.new(self.size, komi=self.komi)

        while True:
            try:
                command_str = input().strip()
            except EOFError:
                return game
            if not command_str:
                continue
            command = GTPCommand.parse(command_str)

            match command.name:
                case "clear_board":
                    game.reset(game.board_size)
                    command.print_output()
                    continue
                case "komi":
                    try:
                        komi = float(command.args[0])
                    except (TypeError, IndexError, ValueError):
                        command.print_output("komi not a float", success=False)
                    else:
                        game.komi = self.komi = komi
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

                    if player != game.next_player:
                        command.print_output(
                            "consecutive moves of the same color are not allowed",
                            success=False,
                        )
                        continue

                    try:
                        move = from_gtp_move(vertex, player)
                    except ValueError:
                        command.print_output("invalid coordinates", success=False)
                        continue

                    if not game.is_valid_move(move):
                        command.print_output("illegal move", success=False)
                        continue

                    game.apply_move(move)
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

                    if player != game.next_player:
                        command.print_output(
                            "consecutive moves of the same color are not allowed",
                            success=False,
                        )
                        continue

                    move = self.bot.select_move(game, player)
                    game.apply_move(move)
                    command.print_output(str(move))
                    if move.is_resign:
                        return game

                    continue

            # other commands are handled manually
            if command.apply(game) == CommandEffect.end_game:
                return game

    def start(self) -> int:
        """Start playing."""

        match self.mode:
            case "ascii":
                game = self.run_ascii()
                print(f"Result: {game.result}")
            case "gtp":
                game = self.run_gtp()
            case _:
                raise ValueError(f"mode not supported: {self.mode}")

        if self.outfile:
            self._save_sgf(game.to_pysgf(), self.outfile)

        return 0
