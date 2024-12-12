import io

import pytest

from mygo.cli import main


def test_help():
    with pytest.raises(SystemExit) as e:
        main(["-h"])
    assert e.value.code == 0


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["-v"])
    assert e.value.code == 0


def test_gtp_commands(monkeypatch, capfd):
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "\n".join(
                [
                    "protocol_version",
                    "name",
                    "version",
                    "known_command known_command",
                    "list_commands",
                    "boardsize 13",
                    "clear_board",
                    "komi 5.5",
                    "play black a1",
                    "play white n13",
                    "play black pass",
                    "genmove white",
                    "genmove black",
                    "showboard",
                    "quit",
                ]
            )
        ),
    )
    out = capfd.readouterr().out
    assert "?" not in out  # '?' stands for error in GTP
    assert main(["--mode", "gtp"]) == 0


class TestBots:
    @pytest.fixture(autouse=True)
    @staticmethod
    def _mock_input(monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("pass\n" * 1000))

    def test_default(self):
        assert main([]) == 0

    def test_random(self):
        assert main(["--bot", "random"]) == 0

    def test_minimax(self):
        assert main(["--bot", "minimax", "--size", "3"]) == 0

    def test_mcts(self):
        assert main(["--bot", "mcts", "--size", "3"]) == 0

    def test_tiny(self):
        assert main(["--bot", "tiny"]) == 0

    def test_small(self):
        assert main(["--bot", "small"]) == 0

    def test_zero(self):
        assert main(["--bot", "zero", "--size", "3", "--bot-args", "10", "--log-level", "debug"]) == 0
