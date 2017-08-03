# DQN-FlappyBird
A toy example of deep reinforcement learning

## Tested with
- python 2.7
- tensorflow
- pygame
- opencv

## How to run
- Clone the repository
- In the root directory of the repository:
```
virtualenv env
```
```
source env/bin/activate
```

- Install all the requirements
```
pip install -r requirements.txt
```

- Run the program
```
python dqn.py
```

## Train the network from scratch
- Delete all the files in the 'model' folder
- In dqn.py, change EBSILON to 0.2
