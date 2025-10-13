# **Reinforcement Learning**

Welcome to my Reinforcement Learning course repository.  
This repository contains all course-related work developed throughout the semester.

---

## **TASK 1**

Simulate a **Grid World** environment, including **reward dynamics** and **policy visualization**.  
Implement **policy iteration** and **value iteration** to find the optimal policy.  
Finally, implement the **Q-learning** algorithm.

All the tasks above are divided into **three notebooks**, where each notebook continues from the previous one.

### **Notebook 1**  
Simulate a **Grid World** environment, including **reward dynamics** and **random policy visualization**.  
[Livrable_1_Majidi_Marouane.ipynb](Task1/Livrable_1_Majidi_Marouane.ipynb)

### **Notebook 2**  
Add the **GridWorld**, **policy iteration**, and **value iteration** implementations, and use them to find the optimal policy.  
[Livrable_2_Majidi_Marouane.ipynb](Task1/Livrable_2_Majidi_Marouane.ipynb)

### **Notebook 3**  
Implement the **Q-learning** algorithm. 

[Livrable_3_Majidi_Marouane.ipynb](Task1/Livrable_3_Majidi_Marouane.ipynb)

---


## **TASK 2**

Continuing with the **Grid World**, we now make the environment more **generalized** by adding parameters that make the **obstacles dynamic** (able to move), as well as allowing the **goal** (terminal state) to change position.

We also implement the **Monte Carlo algorithm** and visualize how the agent’s behavior evolves with respect to the **grid size**.

Additionally, we activate the **dynamic goal** feature to test whether the agent truly understands the concept of the goal.  
We observe that it doesn’t fully learn what a goal is — highlighting the need to explore **other reinforcement learning algorithms**.
[Livrable_4_Majidi_Marouane.ipynb](Task2/RL_MC.ipynb)

--

After observing the limitations of the algorithm when the goal position changes,  
we recognize the need for more advanced methods such as **Neural Network Q-Learning**.  

In this approach, the neural network learns a **general representation of the “goal”**, which demonstrates the **advantage of function approximation** over traditional tabular methods.  
[Livrable_5_Majidi_Marouane.ipynb](Task2/NNQ-learning.ipynb)

---

## **TASK 3: Pacman**
In this task, we solved the Pacman Reinforcement Learning Project (Berkeley Project 3).
We implemented and tested different reinforcement learning agents by updating the following files:

- valueIterationAgents.py
- qlearningAgents.py
- analysis.py

[Check the Folder](./Task3(Pacman)/reinforcement)

--

And then, we compared three models: the standard Q-Learning Agent, an ε-greedy Q-Learning Agent that balances exploration and exploitation, and an Approximate Q-Learning Agent that learns weights for state features to generalize across similar states.

| **Agent Type**           | **Training Episodes** | **Average Reward** |
|:-------------------------:|:---------------------:|:------------------:|
| Approximate Q-Learning   | 2000 / 2000           | **441.40**         |
| ε-Greedy Q-Learning      | 2000 / 2000           | **-170.89**        |
| Standard Q-Learning      | 2000 / 2000           | **216.05**         |

[Comparison Folder](./Task3(Pacman)/reinforcement/Comparison)
