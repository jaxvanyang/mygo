import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import mygo
from mygo.game.types import Game, Move, Point


class CommandEffect(Enum):
    """The effect to the game of applying a command."""

    no_effect = 0
    next_round = 1
    end_game = 2


class ASCIICommand:
    "ASCII mode command."

    known_commands = ("help", "exit", "quit", "pass", "resign", "save", "move")
    help_msg = """Commands:
    help\t\tDisplay this help menu
    exit\t\tExit MyGo
    quit\t\tExit MyGo
    pass\t\tPass on your move
    resign\t\tResign the current game
    save <file>\t\tSave the current game
    <move>\t\tA move of the format <letter><number>
    """

    def __init__(self, command_name: str, arg: Any = None) -> None:
        self.name = command_name
        self.arg = arg

    @staticmethod
    def parse_point(gtp_coords: str) -> Point | None:
        """Create a point from GTP coordinates.

        Return None if cannot parse.
        """

        gtp_coords = gtp_coords.upper()
        if match := re.match(f"([{Move._COLS}])([\\d]+)", gtp_coords):
            return Point(int(match[2]), Move._COLS.index(match[1]) + 1)
        else:
            return None

    @classmethod
    def parse(cls, command_str: str):
        """Parse a command from the input string.

        Return None if it's invalid.
        """
        command_list = command_str.split()
        if not command_list:
            return None

        command, *args = command_list

        for c in cls.known_commands[:-1]:
            if c.startswith(command):
                command = c
                break

        if args:
            if len(args) == 1 and command == "save":
                return cls(command, Path(args[0]))
            return None

        if command in cls.known_commands[:5]:
            return cls(command)

        if point := cls.parse_point(command):
            return cls("move", point)

        return None

    def apply(self, game: Game) -> CommandEffect:
        """Apply this command to game.

        Return the effect to the game of applying this command.
        """

        match self.name:
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
                logging.getLogger("mygo").warning(
                    "Applying save command to game has no effect"
                )
                return CommandEffect.no_effect
            case "move":
                game.apply_move(Move.play(self.arg))
                return CommandEffect.next_round

        raise RuntimeError(f"Unexpected command: {self.name}")


class GTPCommand:
    known_commands = (
        "protocol_version",
        "name",
        "version",
        "known_command",
        "list_commands",
        "quit",
        "boardsize",
        "clear_board",
        "komi",
        "play",
        "genmove",
        "showboard",
    )

    def __init__(
        self, command_name: str, id: int | None = None, args: list[str] | None = None
    ) -> None:
        self.id = id
        self.name = command_name
        self.args = args

    def print_output(self, output: str = "", success: bool = True) -> None:
        """Print command output."""

        prompt = "=" if success else "?"
        id_str = str(self.id) if self.id is not None else ""
        print(f"{prompt}{id_str} {output}\n")

    @classmethod
    def parse(cls, command_str: str) -> "GTPCommand":
        """Parse GTP command."""

        command_list = command_str.split()
        if not command_list:
            return cls("unknown")

        try:
            command_list[0] = int(command_list[0])
        except ValueError:
            pass

        match command_list:
            case [name] if name in cls.known_commands:
                return cls(name)
            case [id, name] if isinstance(id, int) and name in cls.known_commands:
                return cls(name, id=id)
            case [name, *args] if name in cls.known_commands:
                return cls(name, args=args)
            case [id, name, *args] if isinstance(
                id, int
            ) and name in cls.known_commands:
                return cls(name, id=id, args=args)

        return cls("unknown")

    def apply(self, game: Game) -> CommandEffect:
        """Apply the command to the game.

        Note: some commands need to be handled manually.
        """
        match self.name:
            case "protocol_version":
                output = "2"
            case "name":
                output = "MyGo"
            case "version":
                output = mygo.__version__
            case "known_command":
                try:
                    output = str(self.args[0] in self.known_commands).lower()
                except (TypeError, IndexError):
                    output = "false"
            case "list_commands":
                output = "\n".join(self.known_commands)
            case "quit":
                self.print_output()
                return CommandEffect.end_game
            case "boardsize":
                try:
                    board_size = int(self.args[0])
                except (TypeError, ValueError, IndexError):
                    self.print_output("boardsize not an integer", success=False)
                    return CommandEffect.no_effect

                if board_size < 1 or board_size > 19:
                    self.print_output("unacceptable size", success=False)
                    return CommandEffect.no_effect

                game.reset(board_size)
                self.print_output()
                return CommandEffect.next_round
            case "clear_board" | "komi" | "play" | "genmove":
                raise RuntimeError(f"Command {self.name} need be handled manually")
            case "showboard":
                output = f"\n{game}"
            case _:
                self.print_output("unknown command", success=False)
                return CommandEffect.no_effect

        self.print_output(output)
        return CommandEffect.no_effect
