# PID Learning Algorithms
Using OpenAI Gym environments to practice training algorithms for PID controllers and simple neural networks.

## Installation
To use the code, create a virtual environment, activate it, and then install the required packages.

Commands for Windows:
```
python -m pip install --user virtualenv
python -m venv env
.\env\Scripts\activate
python -m pip install -r requirements.txt
```

Commands for Unix/macOS:

```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

## Running scripts

First make sure the virutal environment is activated.

Commands for Windows:

```
.\env\Scripts\activate
```

Commands for Unix/macOS:

```
source env/bin/activate
```

After that you can run any of the scripts inside gyms by running the following commands.

Commands for Windows:

```
python Cart_Pole.py
python Mountain_Car.py
python Pendulum.py
```

Commands for Unix/macOS:

```
python3 Cart_Pole.py
python3 Mountain_Car.py
python3 Pendulum.py
```
