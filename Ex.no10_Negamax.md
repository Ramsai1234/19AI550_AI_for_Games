# Ex.No: 10  Implementation of Negamax Search
### DATE: 03/10/2024                                                                           
### REGISTER NUMBER : 212221240041
### AIM: 
Write a Nega-max search algorithm to find the root value of Player from the  graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Every level, score is evaluated by negamax function 
8. Call the negamax function  and print the optimum value of Max player.
9. Stop the program. 

### Program:
```
import math

def negamax(curDepth, nodeIndex, scores, targetDepth):
    # Base case: target depth reached
    if curDepth == targetDepth:
        return scores[nodeIndex]

    # Negamax assumes max turn is represented by positive values
    value1 = negamax(curDepth + 1, nodeIndex * 2, scores, targetDepth)
    value2 = negamax(curDepth + 1, nodeIndex * 2 + 1, scores, targetDepth)

    return max(-value1, -value2)  # Flip the sign for the other player's turn

# Driver code
scores = [3, 5, 2, 9, 12, 5, 23, 20]
treeDepth = math.log(len(scores), 2)  # Calculate depth of node, log(8, base 2) = 3
print("The optimal value is: ", end="")
print(negamax(0, 0, scores, int(treeDepth)))
```

### Output:

![negamax](https://github.com/user-attachments/assets/20672275-38e1-47a2-b6e5-0366a3cddabf)


### Result:
Thus the root value of player was found using negamax search.
