# Federated Learning and Differential Privacy using TensorFlow and Flower

This project demonstrates:

1. Basic Federated Learning (FL)
2. Federated Learning with Differential Privacy (FL + DP)

The implementation uses:

- TensorFlow for model training
- Flower for federated orchestration
- TensorFlow Privacy for Differential Privacy (DP-SGD)

A MobileNetV2 model is trained on the CIFAR-10 dataset.


---

## Requirements

- Python 3.10.0
- pip

This project was tested with **Python 3.10.0**.

---

## Step 1 — Create Virtual Environment (Python 3.10)

Make sure Python 3.10 is installed.

Check version:

```bash
python --version
```

Create virtual environment:

```bash
python -3.10 -m venv venv
```

---

## Step 2 — Activate Virtual Environment

```bash
venv\Scripts\activate
```
---
## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```
---

Running Federated Learning (FL)

The federated pipeline consists of:

One server

One or more clients

## Step 4 — Start Server

Open a terminal:
```bash
python server.py
```

## Step 5 — Start Client

Open another terminal (with the same virtual environment activated):

**For FL Client**
```bash
python cl.py
```
**For FL-DP Client**
```bash
python cl_dp.py
```
You can simulate multiple clients by running them in multiple terminals