# MyGo

My BetaGo implementation in PyTorch!

## References

- The book: [*Deep Learning and the Game of Go*](https://www.manning.com/books/deep-learning-and-the-game-of-go),
  aka 《深度学习与围棋》.
- The repos: [BetaGo](https://github.com/maxpumperla/betago), [Code and other material for the book](https://github.com/maxpumperla/deep_learning_and_the_game_of_go).

## Notes

- [Chapter 7 training reproducibility](https://github.com/maxpumperla/deep_learning_and_the_game_of_go/issues/108).
- [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
  expects the input to contain the **unnormalized** logits.
