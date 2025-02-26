CT-BaB: Certified Training with Branch-and-Bound
==================

[![Arxiv](https://img.shields.io/badge/arXiv-2411.18235-B31B1B.svg?logo=arxiv)](https://arxiv.org/abs/2411.18235)

CT-BaB is a generally formulated framework of verification-aware neural network
training (a.k.a. certified training) with **training-time branch-and-bound**.
It aims to **train verification-friendly neural networks with stronger verifiable guarantees**
that hold for an entire input region-of-interest.
As a case study, the framework has been demonstrated on an application of
learning Lyapunov-stable neural network-based controllers in nonlinear
dynamical systems, where the controllers can be verified to satisfy the
Lyapunov asymptotic stability within a region-of-attraction.
On the 2D quadrotor dynamical system, verification for our model
is more than 5X faster (i.e., the model is more verification-friendly),
compared to [the previous state-of-the-art work](https://arxiv.org/pdf/2404.07956),
while the volume of our region-of-attraction is 16X larger
(i.e., the model achieves stronger verifiable guarantees).

## Setup

Python 3.11+ is recommended.

Install dependencies:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA; pip install -e .)
(cd alpha-beta-CROWN/complete_verifier; pip install -r requirements.txt)
pip install -r requirements.txt
```

## Training

Run `train.py` with arguments.

### Main arguments

`dir` specifies the directory for the output (checkpoints, etc.).

Problem and data:
* `dynamics`: name of the dynamical system, imported in [models/control.py](models/control.py).
* `upper_limit`: a list of values to specify the region-of-interest on each input dimension.
* `max_init_size`: maximum size of each initial sub-region.

Modeling:
* `lyapunov_func`: Type of Lyapunov function
(`quadratic` denotes quadratic Lyapunov function and `nn` denotes NN Lyapunov function).
* `lyapunov_R_rows`: If a quadratic Lyapunov is used,
it specifies the number of rows in the parameter of the quadratic function.
* `lyapunov_width`: If a NN Lyapunov is used, it specifies the width of the NN.
* `lyapunov_depth`: If a NN Lyapunov is used, it specifies the depth of the NN.
* `controller_width`: Width of the controller network.
* `controller_depth`: Depth of the controller network.

Region-of-attraction (ROA):
* `rho_ratio`: Ratio of sampled data points that are desired to be within ROA
(for controlling the ROA size).
* `rho_penalty`: Strength of the regularization for ROA size.

Training hyperparameters:
* `batch_size` and `lr`.

### Example

Here is a example of training on the 2D quadrotor system:
```bash
NAME=example_model
DIR=./$NAME
python -u train.py --dir $DIR \
--batch_size 30000 --lr 5e-3 \
--dynamics quadrotor2d \
--upper_limit 0.75 0.75 1.57 4 4 3 --max_init_size 0.8 \
--rho_ratio 0.1 --rho_penalty 0.1 \
--lyapunov_func quadratic --lyapunov_R_rows 6 \
--controller_width 8 --controller_depth 2
```

(More examples to be added.)

## Testing (verification)

After a model is trained,
we use our [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
toolbox for verification, which is configured to compute verified bounds by
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)
enhanced with input-space branch-and-bound.
You may check the documentation of α,β-CROWN and auto_LiRPA to learn more about
their usage and background.

### Step 1: Generate specifications for the verification

Specifications are generated in the [VNN-LIB](https://www.vnnlib.org/) format,
similar to benchmarks in [VNN-COMP](https://sites.google.com/view/vnn2024).

Basic usage:
```bash
python generate_vnnlib.py SPEC_PATH -u UPPER_LIMITS -v VALUE_LEVELSET
```

* `SPEC_PATH` is a prefix for the path of specification files.
* `-u UPPER_LIMITS` is a list of values that specify the region-of-interest,
similar to that in the training.
* `-v VALUE_LEVELSET` is for specifying the level set of ROA.
Since 1.0 is used in training by default,
we may set a value slightly smaller than 1.0 (e.g., 0.9) for testing.

Example:
```
python generate_vnnlib.py specs/quadrotor2d \
-u 0.75 0.75 1.57 4 4 3 -v 0.9
```

### Step 2: Running verification

Enter α,β-CROWN to run the verification with the specifications we have generated.

Basic usage for running the verification:
```bash
export TRAINING_PATH=$(pwd)
cd alpha-beta-CROWN/complete_verifier

PYTHONPATH=$TRAINING_PATH python -u abcrown.py \
--config YAML_CONFIG_FILE \
--load_model MODEL_FILE \
--csv_name SPEC_PATH.csv \
--batch_size BATCH_SIZE
```

* `YAML_CONFIG_FILE` is a [YAML configuration file](https://github.com/Verified-Intelligence/alpha-beta-CROWN?tab=readme-ov-file#instructions) required for running α,β-CROWN.
We have included the configuration files for the systems
we used in the [verification](./verification) folder.
Note that if you modify the models (architecture or hyperparameters),
you may need to change the configuration files as well.
* `MODEL_FILE` is the checkpoint to be verified.
* `SPEC_PATH.csv` is a .csv file which lists instances for verification.
Here `SPEC_PATH` is the same as the one you specify for generating specification files.
* `BATCH_SIZE` is the batch size for test-time branch-and-bound,
and it is recommended to use a value as large as possible,
as long as it fits the GPU memory.

Example:
```bash
export TRAINING_PATH=$(pwd)
cd alpha-beta-CROWN/complete_verifier

PYTHONPATH=$TRAINING_PATH python -u abcrown.py \
--config $TRAINING_PATH/verification/quadrotor2d.yaml \
--load_model $TRAINING_PATH/example_model/10000.pt \
--csv_name $TRAINING_PATH/specs/quadrotor2d.csv \
--batch_size 3000000
```

Optionally, you may pipe the output of verification to a file
by appending `| tee LOG_FILE.txt` to the command.

## Reference

```bibtex
@article{shi2024ctbab,
  title={Certified Training with Branch-and-Bound: A Case Study on Lyapunov-stable Neural Control},
  author={Shi, Zhouxing and Hsieh, Cho-Jui and Zhang, Huan},
  journal={arXiv preprint arXiv:2411.18235},
  year={2024}
}
```