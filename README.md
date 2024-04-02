# 机器学习项目：剪刀石头布 (Rock Paper Scissors)

欢迎来到剪刀石头布人工智能项目！这是一个基于 Python 实现的剪刀石头布游戏，专门为了展示在 FreeCodeCamp 的《Machine Learning with Python》课程中完成的『[Rock Paper Scissors][fcc-ml-rps]』项目。通过这个项目，我理解了机器学习基础概念，利用了一些简单但有效的技术来实现AI对手。

## 双 马尔可夫链(Markov chain) 决策

你可能会注意到，这个项目中选择使用马尔可夫链而不是循环神经网络（RNN）。

* 问题的简单性： 剪刀石头布游戏是一个非常简单的问题，其规则和模式相对固定且易于理解。
* 数据的结构化：  剪刀石头布游戏的数据具有结构化的特点，因为每一轮游戏的结果都可以归纳为对手上一步的动作和当前玩家的动作。马尔可夫链适用于处理这种结构化的序列数据，能够很好地捕捉到状态之间的转移概率。
* 采用两个马尔可夫链来决策适应不同的对手。 一个正向的Markov用来预测对手，另一个用来反向检测是否被对手预测（预判对手的预判）。
* 在实现过程中，我经历了多次参数调谐，以找到最优的参数配置，以提高AI的性能和稳定性。经过不懈的努力和实验，我成功地找到了一组最佳参数，使得AI在各种情况下都表现出色。

## 版权声明

该项目采用 MIT License 授权。 

本README由项目作者帮助CHATGPT生成。


[fcc-ml-rps]: https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors