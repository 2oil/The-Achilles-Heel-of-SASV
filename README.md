
# The Achilles’ Heel of Spoofing-Aware Speaker Verification: A Module-Level Analysis of Adversarial Vulnerabilities

This repository contains the official implementation of the adversarial attack framework proposed in:

> **“The Achilles’ Heel of Spoofing-Aware Speaker Verification: A Module-Level Analysis of Adversarial Vulnerabilities”**  
> Yowon Lee, Thien-Phuc Doan, Sanghyun Hong, and Souhwan Jung  
> *PeerJ Computer Science, under review*  

---

## 🔍 What Is This Repository?

Spoofing-Aware Speaker Verification (SASV) systems combine:

- **ASV (Automatic Speaker Verification)** – verifies whether the speaker matches the enrolled identity  
- **CM (Spoofing Countermeasure)** – detects whether the speech is bona fide or spoofed

While recent SASV systems are robust against *spoofing* attacks, their robustness against **adversarial examples** has been far less explored.

This repository provides a **module-level adversarial attack framework** that can:

- Target **only the CM module** (spoofing detector), or  
- Target **only the ASV module** (speaker verifier),

and then compare how vulnerable each module is under carefully controlled attack settings.

---

## 🧪 Supported Attack Scenarios

We consider two primary attack scenarios:

| Module Targeted | Objective                                                |
|-----------------|----------------------------------------------------------|
| **CM module**   | Make spoofed samples be misclassified as *bona fide*     |
| **ASV module**  | Make non-target impostors be accepted as the *target*    |

Each scenario is implemented using standard gradient-based attacks:

- **FGSM** – Fast Gradient Sign Method  
- **BIM** – Basic Iterative Method  
- **PGD** – Projected Gradient Descent  

The key idea of this framework is to **isolate the attack target (ASV vs. CM)** while keeping the rest of the pipeline fixed, so that the adversarial robustness of each module can be compared fairly.

---

## 📂 Datasets

### 1. ASVspoof 2019 LA

- Dataset : [https://huggingface.co/datasets/LanceaKing/asvspoof2019](https://huggingface.co/datasets/LanceaKing/asvspoof2019)
- Place the evaluation audio files under:

```text
./LA/ASVspoof2019_LA_eval/flac/
````

### 2. ASVspoof 5

* Dataset: [https://huggingface.co/datasets/jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5)
* Place the evaluation audio files under:

```text
./ASVspoof5/release_eval/flac_E_eval
```

### 3. Metadata and Protocols

* The metadata used in the paper is stored under:

```text
./PeerJ/protocol/
    ├─ ASVspoof2019.csv
    └─ ASVspoof5.csv
```

* The notebook for sampling the evaluation subset and setting random seeds is:

```text
./PeerJ/protocol/protocol.ipynb
```

---

## 🧠 Models

This framework assumes pretrained **CM** and **ASV** models. You can use the following open-source implementations:

### Countermeasure (CM) Models

* **AASIST**
  [https://github.com/clovaai/aasist.git](https://github.com/clovaai/aasist.git)
* **AASIST-SSL**
  [https://github.com/issflab/ssl-antispoofing.git](https://github.com/issflab/ssl-antispoofing.git)
* **RawNet2**
  [https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/README.md](https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/README.md)

### Automatic Speaker Verification (ASV) Models

* **ECAPA-TDNN**
  [https://github.com/TaoRuijie/ECAPA-TDNN.git](https://github.com/TaoRuijie/ECAPA-TDNN.git)
* **ResNet34v2 (Joint-Optimized SASV)**
  [https://github.com/eurecom-asp/sasv-joint-optimisation.git](https://github.com/eurecom-asp/sasv-joint-optimisation.git)
* **NeXt-TDNN**
  [https://github.com/dmlguq456/NeXt_TDNN_ASV.git](https://github.com/dmlguq456/NeXt_TDNN_ASV.git)

Download the corresponding pretrained checkpoints and either:


* and update the checkpoint paths directly in:

  * `attack.sh`
  * `gen_ad_asv.py`
  * `gen_ad_cm.py`

---

## ▶️ Running Attacks

We provide a unified launcher script, **`attack.sh`**, which automatically dispatches to the appropriate attack script based on the **prefix** of the attack method.

* If the method name begins with `CM`, it calls **`gen_ad_cm.py`**
* If it begins with `ASV`, it calls **`gen_ad_asv.py`**

### 1. Default Configuration (attack.sh)

Example default settings:

```bash
batch_size=1
input_path='./LA/ASVspoof2019_LA_eval/flac/'
output_path='./'
adv_method1='CM_FGSM_0001'  # or 'ASV_BIM_0003'
seed='251'
```

* `adv_method1` encodes:

  * the **target module** (`CM` or `ASV`),
  * the **attack type** (`FGSM`, `BIM`, `PGD`),
  * and often the **epsilon** or step size suffix (e.g., `0001`).

> 💡 **Reproducibility**
> In the paper, we run each experiment three times with seeds **251, 252, 253** and report the aggregated statistics.

### 2. Automatic Script Selection

Inside `attack.sh`, we automatically route to the correct generator:

```bash
if [[ "$adv_method1" == CM* ]]; then
    script="gen_ad_cm.py"
elif [[ "$adv_method1" == ASV* ]]; then
    script="gen_ad_asv.py"
else
    echo "❌ Error: adv_method1 must start with 'CM' or 'ASV'"
    exit 1
fi
```

The resulting command is:

```bash
CUDA_VISIBLE_DEVICES=0 python ${script} \
    --batch_size ${batch_size} \
    --input_path ${input_path} \
    --output_path ${output_path} \
    --adv_method1 ${adv_method1} \
    --seed ${seed}
```

### 3. Run the Attack

Simply execute:

```bash
bash attack.sh
```


---

## 📊 Evaluation

For evaluation, this repository provides ASV and CM scoring utilities under:

```text
./eval/
```

### 1. Module-Level Scores

* Use the scripts in `./eval` to compute **score files** for each module:

  * **CM-only** scores (e.g., for AASIST, AASIST-SSL, RawNet2)
  * **ASV-only** scores (e.g., for ECAPA-TDNN, ResNet34v2, NeXt-TDNN)

These are required to compute the metrics for:

* **SFv1, SFv2, C-c, C-a**: setups that need **module-wise scores**.

### 2. SASV Systems (JO, DF)

For joint SASV systems such as:

* **JO (Joint Optimization)**
* **DF (Dynamic Fusion)**

you can download the pretrained models from:

* [https://github.com/eurecom-asp/sasv-joint-optimisation.git](https://github.com/eurecom-asp/sasv-joint-optimisation.git)

### 3. Computing EER and a-DCF

After generating all necessary score files, you can reproduce the metrics in the paper using:

```text
./PeerJ/origin/eval.ipynb
```

This notebook computes:

* **EER (Equal Error Rate)**
* **a-DCF (minimum normalized tandem detection cost function)**

for both **clean** and **adversarial** conditions, matching the evaluation protocol described in the manuscript.

---

## 🖼️ Example Architecture

An overview of the proposed SASV attack pipeline is illustrated below:

![SASV Attack Pipeline](figures/fig1.png)

The figure shows how the framework injects adversarial perturbations at the waveform level while monitoring the responses of the **CM**, **ASV**, and **SASV fusion** modules independently.

---

## 💡 Tips & Notes

* `adv_method1` **must** start with `CM` or `ASV` to trigger the correct attack script.
* You can easily extend this framework to:

  * new **attack variants** (e.g., targeted attacks),
  * new **fusion strategies**,
  * or new **CM/ASV backbones** by plugging them into the same interface.
* For exact reproducibility of the paper:

  * Use the **same seeds** (`251, 252, 253`),
  * The provided **protocol CSVs**,
  * And the **evaluation notebook** in `./PeerJ/origin/`.

---

## 📬 Contact

If you have questions, need the PDF, or are interested in collaboration, feel free to reach out:

**Yowon Lee**
✉️ `agent251@soongsil.ac.kr`
