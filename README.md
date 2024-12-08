# MyGo

> It's my Go bot!

Experimental Go bot implemented in PyTorch.

PS: This project has nothing to do with [MyGO!!!!!](https://en.wikipedia.org/wiki/BanG_Dream!#Introduction_of_MyGO!!!!!_and_Ave_Mujica_(2022%E2%80%93present)),
except its name.

## References

- Mainly the book: [*Deep Learning and the Game of Go*](https://www.manning.com/books/deep-learning-and-the-game-of-go),
  Chinese version named 《深度学习与围棋》.
- The repos: [BetaGo](https://github.com/maxpumperla/betago), [Code and other material for the book](https://github.com/maxpumperla/deep_learning_and_the_game_of_go).
- The papers:
	- AlphaGo: [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961).
	- AlphaGo Zero: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270).
- [GNU Go](https://www.gnu.org/software/gnugo): the CLI design reference.
- [SGF file format FF[4]](https://www.red-bean.com/sgf).
- [GTP - Go Text Protocol](https://www.lysator.liu.se/~gunnar/gtp).
- [Tromp-Taylor rules](https://tromp.github.io/go.html).
- [Monte Carlo tree search - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

## Notes

- [Chapter 7 training reproducibility](https://github.com/maxpumperla/deep_learning_and_the_game_of_go/issues/108).
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
  expects the input to contain the **unnormalized** logits, which is different from Keras.
