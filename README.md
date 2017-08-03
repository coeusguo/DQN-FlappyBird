# DQN-FlappyBird
A toy implementation of deep reinforcement learning based on [DQN Nature](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


## Tested with
- Ubuntu 16.04
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
