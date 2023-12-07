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
* ``scp`` the ``src`` and ``*.pbs`` files to the login node and make sure all the data is copied to the data folder ``TODO``
* ``$ gsub install_venv.pbs -v cluster=joltik``, to connect to the GPU cluster called ``joltik`` and create a virutal environment called ``venv_joltik``. This will also install all the required Python packages that are listed in the ``requirements.txt`` file.

### Running the scripts:
* As mentioned the directory ``$VCS_DATA/ml`` is used as the location from which the scripts are executed. Run the code by invoking ``gsub run_experiment.pbs -v cluster=joltik``.
* Follow stats by ``qstat [PID]``
* Output is stored in ``TODO``