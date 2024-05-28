import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from mygo.game.types import Game, Move, Point


class CommandEffect(Enum):
    """The effect to the game of applying a command."""

    no_effect = 0
    next_round = 1
    end_game = 2


class ASCIICommand:
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
        command_list = command_str.split()
        if not command_list:
            return None

        command, *args = command_list

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
                logging.getLogger("mygo").warning(
                    "Applying save command to game has no effect"
                )
                return CommandEffect.no_effect
            case "move":
                game.apply_move(Move.play(self.arg))
                return CommandEffect.next_round

        raise RuntimeError(f"Unexpected command: {self.type}")
