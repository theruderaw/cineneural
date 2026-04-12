# Movie Recommendation System with NLP Analysis
> **Project Status**: Completed &bull; **Backend**: Python/FastAPI &bull; **NLP**: TF-IDF &bull; **UI**: Tailwind CSS

---

## 🏛️ Block 1: ETL Pipeline & Feature Engineering (The Metadata Pipeline)
The system’s performance is built upon a high-fidelity Extract-Transform-Load (ETL) pipeline designed to resolve **Lexical Sparsity** in film metadata.

### 1.1. Resolving Categorical Sparsity
Global film datasets are inherently sparse in their categorical tagging. Our pipeline implements a **zero-signal imputation strategy** for the five primary metadata columns ($C$):
$$C \in \{\text{Plot, Genre, Director, Cast, Origin/Ethnicity}\}$$

Unlike standard imputation, we utilize empty-string injection to ensure that cosine similarity calculations ignore missing fields without introducing statistical bias:
```python
for col in ['Genre', 'Director', 'Cast', 'Plot', 'Origin/Ethnicity']:
    df[col] = df[col].fillna('')
```

### 1.2. Semantic Integrity in Genre Tokenization
Standard whitespace tokenization fails with compound genres (e.g., *"Science Fiction"*). We implement a **Comma-Delimited Tokenization Engine** to preserve semantic units as single atoms in the Latent Space:
```python
def genre_tokenizer(text):
    return [x.strip() for x in str(text).split(',')]
```

---

## 🧠 Block 2: Statistical NLP & N-Gram Dynamics (The Bigram Latent Space)
To solve the "Context Blindness" of unigram models, the engine maps narratives into a **Bigram-enabled TF-IDF Vector Space**.

### 2.1. Signal-to-Noise Ratio (SNR) Optimization
The semantic density of a movie plot is highly variable. We optimize the **Signal-to-Noise Ratio** by applying frequency-based filtering to the latent features ($f$):
*   **Lower Bound ($min\_df=5$):** Removes lexicographical noise and unique identifiers.
*   **Upper Bound ($max\_df=0.8$):** Prunes "Stop Words" and generic narrative tokens that dilute the specific thematic signal.

### 2.2. Mathematical Foundation of Bigrams
The weight $w$ of a bigram $b$ in a movie's plot $P$ is calculated via:
$$w(b, P) = \text{TF}(b, P) \cdot \log\left(\frac{N}{\text{DF}(b)}\right)$$
By utilizing $n \in \{1, 2\}$, the system captures complex cinematic concepts like *"Deep Space"*, *"Cold War"*, or *"Star Cross'd"* as unified semantic vectors, significantly reducing **Latent Ambiguity**.

---

## ⚙️ Block 3: The Inference Engine (MDLF Optimality)
The core logic utilizes a **Multi-Dimensional Latent Fusion (MDLF)** architecture, addressing the "Black Box" transparency problem in Neural Collaborative Filtering.

### 3.1. Similarity Algebra (Cosine Proximity)
Each movie $m$ is treated as a point in a high-dimensional hyperspace. Similarity between a query movie $Q$ and a candidate $C$ is calculated across six independent signals ($S_i$):
$$\text{Sim}(Q, C)_i = \frac{\mathbf{v}_{Q,i} \cdot \mathbf{v}_{C,i}}{\|\mathbf{v}_{Q,i}\| \|\mathbf{v}_{C,i}\|}$$

### 3.2. Temporal Era & Squared Decay ($S_y$)
Unlike linear metrics, we implement **Exponential Decade Sensitivity** to ensure that films from the same decade are prioritized. The similarity is calculated via a bounded linear difference normalized by a century-constant ($100$), then **squared** to create a non-linear decay:
$$S_y = \max\left(1 - \frac{|\text{Year}_Q - \text{Year}_C|}{100}, 0\right)^2$$
This transformation effectively clusters films within their specific cinematic era by rewarding proximity exponentially.

### 3.3. Weighted Fusion Formula ($NLF$)
The final ranking is determined by the **Normalized Linear Fusion ($NLF$)** of these signals:
$$S_{final} = \sum_{i \in \{p, g, d, o, y, c\}} w_i \cdot S_i$$

---

## 🧬 Block 4: Symbolic Cognitive Layer (Heuristic Grounding)
To prevent **"Bag-of-Themes Collapse"**—where statistical keyword matches lose higher-level narrative context—the system implements a **Symbolic Heuristic Layer**.

### 4.1. Narrative Anchors (Cognitive Buckets)
We define $K$ symbolic anchors representing deep cinematic tropes. The system detects the **Conceptual Intensity** of a movie by matching its plot against these curated keyword dictionaries:
*   **Celestial Group:** $\{ \text{galaxy, star, cosmic, relativity, void} \}$
*   **Legacy Group:** $\{ \text{dynasty, family, inheritance, childhood} \}$

### 4.2. Heuristic Bonus logic
If the query movie $Q$ and candidate $C$ share a dominant Conceptual Bucket, a **1.08x Thematic Depth Multiplier** is applied:
```python
if primary_query_bucket == primary_candidate_bucket:
    final_score *= 1.08  # Thematic Depth Bonus
```

---

## 🕵️ Block 5: Explainability & Interpretability (XAI Framework)
True AI systems must be "Interpretable by Design." CineNeural implements a **Post-Hoc Feature Attribution** framework.

### 5.1. Feature Importance Analysis
For each recommendation, the system performs an internal audit of the MDLF inputs to extract the **Maximum Contributor ($arg \max$)**:
$$T = \arg \max_{i \in \{p, g, d, o, y, c\}} (w_i \cdot S_i)$$

### 5.2. Interpretability-Performance Trade-off
While deep neural nets offer marginally higher density, they sacrifice transparency. Our architecture chooses **Linear Multi-Dimensional Fusion** specifically because it provides **Global Interpretability**, allowing the user to precisely understand why *Contact* was recommended over *Gravity*.

---

## 🚀 Appendix: Rapid Deployment & Setup

### 1. Clone & Synchronize
```bash
git clone https://github.com/theruderaw/cineneural.git
cd cineneural
```

### 2. Environment Initialization
```bash
# Create and activate a dedicated virtual environment
python3 -m venv venv
source venv/bin/activate

# Synchronize dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Engine Cold-Start
```bash
# Initialize the Latent Space matrices
python3 train_model.py

# Launch the asynchronous inference server
uvicorn main:app --reload
```

---
*Created for Advanced AIML & NLP Technical Submission • 2026*
