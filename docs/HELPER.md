# Helper Commands

This document collects frequently used commands for working with the remote server and monitoring experiments.

## Screen Sessions

```bash
screen [-ls] [-r]
```

## Process Monitoring

```bash
htop
```

## Server Connection

```bash
ssh -l AP09650 replica.logti.etsmtl.ca
```

## Activate Conda Environment

```bash
source miniconda3/bin/activate && conda activate QAML && cd QMLBenchmark
```

## Copy Results from Remote to Local Machine

```bash
scp -r AP09650@replica.logti.etsmtl.ca:QMLBenchmark\logs C:\Users\fwilh\Desktop\REPLICA\

scp -r C:\Users\fwilh\Desktop\Experiments AP09650@replica.logti.etsmtl.ca:Backup
```

## Run Job in the Background

```bash
nohup python QMLBenchmark/src/main.py > funsies.out 2> oopsies.err < /dev/null &
```

## Follow Job Output

```bash
tail -f funsies.out
```

## Watch Training Locally

```bash
tensorboard --logdir=.
```

## Watch Training Remotely

1. Run those commands:

    ```bash
    ssh -L 6006:localhost:6006 AP09650@replica.logti.etsmtl.ca
    source miniconda3/bin/activate && conda activate QAML && cd QMLBenchmark
    tensorboard --logdir=. --port=6006
    ```

2. Open a browser on your local machine and navigate to:

    ```
    http://localhost:6006
    ```

## Update pip packages

    ```bash
    pip --disable-pip-version-check list --outdated --format=json | python -c "import json, sys; print('`n'.join([x['name'] for x in json.load(sys.stdin)]))" | ForEach-Object { pip install -U $_ }
    ```


    ```bash
    pip --disable-pip-version-check list --outdated --format=json | python -c "import json, sys; print('\n'.join([x['name'] for x in json.load(sys.stdin)]))" | xargs -n1 pip install -U
    ```

## Profiling

scalene ./src/train.py

--cpu --gpu --memory for profiling all the different things to profile

--profile-all to profile external libs instead of just getting python/native
--profile-all --profile-only pennylane,torch,lightning,pytorch_lightning,numpy

HTML report written where you choose:
scalene --cli --html --outfile /path/to/reports/profile.html your_program.py

JSON instead (useful for automation):
scalene --cli --json --outfile /path/to/reports/profile.json your_program.py