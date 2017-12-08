# reinforcement_learning_pytorch
some basic implementations of algorithms in reinforcement learning in pytorch.



## Submit Dairy

---20171120---

try to use a q-function &  td-learning to do this task. 

FAILED.

---20171120---

solve the sarsa problem in some way.
A trick use here: sarsa is an online algorithm. Everytime when we update the parameter, a batch only have one sample, which may lead this network to unconvergence.

Two ways to solve this problem:

1. Change the optimization function

2. Sum up the loss of several steps. In this updated version, I sum up the losses of sive steps, and back propogate them together.

---20171121---

Made a mistake yesterday...

The Sarsa algorithm doesn't have any random action-picking($\ecsilon$ process...
I also implement a td-q learning model...

reformulate the structure of the code. Add the algorithms I want to implement in three weeks. :) 

come on! becky!

---20171122---

Slow speed...

Finish REINFORCE part.

The output of a policy netowrk should be softmax...

Monte Carlo method is much more stable. In fact, I didn't figure out which factor lead to it... policy gradient or MC?

next step is ac and trpo

---20171126---

---20171201---

Finish AC-TD RL(off-line)\\
much more stable than the former version.


---20171208---

Trying a new model "pendulum" for several days without a very good result. DDPG seems to be a working model, but I am still busy on tuning the the parameters to make it work...

---20171208v2---

Figure out the problem !

When intilize environment, I use gym.make("name").unwrapped. In this situtaion, the pendulum won't stop util we get to the maximum step. "wrapper" helps us access the inner environment. How it works here is still unclear.
