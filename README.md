# Learning contracting control for a network of homogeneous Lorenz attractors
Code for reproducing experiments of Giaccagli et al. ["Synchronization in networks of
nonlinear systems: Contraction analysis via Riemannian metrics and deep-learning for feedback
estimation"](https://ieeexplore.ieee.org/document/10541048), section V.

## **Project Overview**

The code consists of three main scripts:

- `find_P.py` – Generates and trains the metric network based on contraction conditions.
- `find_alpha.py` – Generates and trains the alpha network based on the integrability condition.
- `test_multiagent.py` – Loads the trained alpha network and evaluates the proposed controller on a network of chaotic Lorenz attractors.

---
## **Usage**

### **Train the Metric Network (`find_P.py`)**

```bash
python find_P.py --net 64,64 --activ relu --dataset_size 10000 --batch_size 128 --n_epochs 50 --learning_rate 0.001 --log_name experiment_1
```

### **Train the Alpha Network (`find_alpha.py`)**

```bash
python find_alpha.py --net 64,64 --activ tanh --dataset_size 10000 --batch_size 128 --n_epochs 50 --learning_rate 0.001 --log_name experiment_2
```

### **Test the Multi-Agent Controller (`test_multiagent.py`)**

```bash
python test_multiagent.py --alpha_path <path-to>/trained_alpha.pth
```

---

## **Command-Line Arguments**

### **Training Scripts (`find_P.py` & `find_alpha.py`)**

| Argument         | Description |
|-----------------|-------------|
| `--net`         | Hidden layer dimensions (comma-separated) |
| `--activ`       | Activation function (`relu` or `tanh`) |
| `--dataset_size` | Total number of samples |
| `--batch_size`  | Batch size for training |
| `--n_epochs`    | Number of training epochs |
| `--learning_rate` | Learning rate (cosine annealing scheduling) |
| `--log_name`    | Log folder name (saved in `runs/`) |

### **Testing Script (`test_multiagent.py`)**

| Argument       | Description |
|---------------|-------------|
| `--alpha_path` | Path to the trained alpha network |

---

## **Results & Logs**

All training logs and model checkpoints are saved in the `runs/` directory.

