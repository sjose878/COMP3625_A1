Write your answers to the following questions in this file, *after* completing everything else.

# 1) How many settings must be evaluated to *exhaustively* search for the best set (e.g. through some kind of brute-force search). State your assumptions and explain how you arrived at your answer. How does your solution compare to this (quantitatively)?



# 2) Why is the 10-knob problem harder than the 2-knob problem? (It may help you to know that the *mechanics* of both problems were the same: the relationship between settings and performance had the same general characteristics, there were just more settings to tune in the 10-knob problem).

It goes from being a 2-dimensional problem to a 10-dimensional problem. The number of different knob settings to try exponentially increases. It takes so much more time and even luck to figure out if one setting change made by the algorithm actually gets us closer to the goal. Algorithms need a lot more time and fine-tuning to navigate a 10-dimensional search space.

# 3) If your program was seen as an "agent program", which of the agent types discussed in class would it be?

If our program was an "agent program," then it would be a utility-based agent. This is because the evaluation of each setting's decode rate score acts much like a utility function. Knob settings are ranked based on the value of their decode rates are, with a higher decode rate being more desirable. This is the performance measure to be maximized by the program. 


# 4) it's often said that the simplest solution is the best. How well would a basic hill-climbing search perform in this problem? Justify your answer using your findings or plots from part 1 (you can answer this question whether or not you used hill-climbing as your approach). 

Assuming a basic hill-climbing search means a single greedy search, it would do horribly. It is very likely to get stuck on a local maxima or a flat plateau and never reach the global maxima. But it is still useful if modified with extra features. In part 2, we used a hill-climb that made stochastic moves to try to escape plateaus and had it restart multiple times with random values so that it had a chance to be "spawned" near the global maxima. Our decode rate for 10 knobs using hill-climbing is not super accurate but it has a fast time complexity.


# 5) When you moved from the 2- to the 10-knob problem, did you change your search algorithm? Why or why not?

Yes, we ended up changing our search algorithm when moving to the 10-knob problem. We found that no matter what settings we tried for gradient descent, the best decode rate found was always extremely low, ranging from 18% to even 0%. This led us to change the algorithm we were using, instead opting for a stochastic hill-climbing approach, which reliably got a decode rate of 60% or more. The main reason we switched to hill-climbing was because it seemed to easy to implement as it is conceptually similiar to gradient descent where you start with an initial state and analyze a neighbor state and choose the most beneficial state. Also it seems that gradient descent when stuck in a plateau would just hopelessly compute gradients which does nothing. Stochastic hill-climbing could escape plateaus with its random jumps. Simulated Annealing was also tested and it got similiar results to hill-climbing but we found it to be less consistent.
