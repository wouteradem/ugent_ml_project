# Machine Learning Project
This project consists of 3 CNN applications:
* Application 1: Classification
* Application 2: Object Detection
* Application 3: Aesthetic Assessment
All the applications are programmed in Python using the PyTorch library.

## Running the code locally
* ``python experiment1.py``

## Running the code in the UGent GPU cluster

### Preparation:
* Create directory ``$VSC_DATA/ml``
* ``scp -r`` copy directly to the directory ``$VSC_DATA/ml``
* From the login node in the ``~/ml`` directory there must be 2 files ``install_venv.pbs`` and ``run_ml.pbs``.
* ``$ qsub install_venv.pbs -v cluster=joltik``, to connect to the GPU cluster called ``joltik`` and create a virutal environment called ``venv_joltik``. This will also install all the required Python packages that are listed in the ``requirements.txt`` file.
* All output is logged in ``~/ml`` directory


### Running the scripts:
* As mentioned the directory ``$VCS_DATA/ml/[APPLICATION]`` is used as the location from which the scripts are executed. Run the code by invoking ``qsub run_ml.pbs -v cluster=joltik``.
* Follow stats by ``qstat [PID]``
* Output is stored in ``TODO``