from mygo.model import Model, SmallModel, TinyModel


def test_get_model():
    assert TinyModel == Model.get_model("tiny")
    assert SmallModel == Model.get_model("small")
