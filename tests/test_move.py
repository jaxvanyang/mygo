from mygo.game import PassMove, Player, PlayMove, Point, ResignMove


class TestEquality:

    def test_play_move(self):
        m1 = PlayMove(Player.black, Point(0, 0))
        m2 = PlayMove(Player.black, Point(0, 0))
        m3 = PlayMove(Player.black, Point(1, 0))
        m4 = PlayMove(Player.white, Point(0, 0))

        assert m1 == m2 and hash(m1) == hash(m2)
        assert m1 != m3
        assert m1 != m4

    def test_pass_move(self):
        m1 = PassMove(Player.black)
        m2 = PassMove(Player.black)
        m3 = PassMove(Player.white)

        assert m1 == m2 and hash(m1) == hash(m2)
        assert m1 != m3

    def test_resign_move(self):
        m1 = ResignMove(Player.black)
        m2 = ResignMove(Player.black)
        m3 = ResignMove(Player.white)

        assert m1 == m2 and hash(m1) == hash(m2)
        assert m1 != m3
