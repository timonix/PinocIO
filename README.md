# ~~~~~~ PinocIO ~~~~~~

## ---------- Windows -----------

### Installation
1. run install_enviroment.bat
it will install and activate a virtual env
2. run install_requirements.bat from inside your venv
installs libs into environment

### Usage
ssh pi@[ip address]

## ---------- Raspberry Pi ----------

### Installation
1. Install virtual environment
2. run command: source .venv/bin/activate
3. Run install_requirements.sh
4. Do the rest

### Usage
1. cd Pinocio
2. source .venv/bin/activate
3. 

## ---------- Pinout ----------

### Motors and servo
M1 is left stepper, M2 is right.

- M1 Enable = GPIO14 
- M1 Step = GPIO15 
- M1 Direction = GPIO18
- M2 Enable = GPIO23 
- M2 Step = GPIO24 
- M2 Direction = GPIO25

- Servo signal = GPIO17

### GPIO Sound
- GPIO 13 (pin33) to A+ 
- GROUND to A-

# ---------- TODO ---------- 

- Make pytorch work on both PC and Pi
- Train gameplay
- Build robot body
  - Motors
  - CAD
  - Servo
  - Controls
- Start working on other parts of the AI

# Training video & reward

1. gather video
2. train video

# Training reward
1. record play
2. lock action net, lock video
3. train reward net

# training action net



