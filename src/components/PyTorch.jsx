import { useState, useMemo } from "react";
import Header from "./Header";
const GROUP_ACCENT = {
  "PyTorch Core":            "#60a5fa",
  "Layer Types":             "#818cf8",
  "Activation Functions":    "#a78bfa",
  "Loss Functions":          "#f87171",
  "Optimizers":              "#fb923c",
  "LR Schedulers":           "#fbbf24",
  "Regularisation":          "#34d399",
  "Data Pipeline":           "#22d3ee",
  "Training Patterns":       "#4ade80",
  "Skorch Core":             "#c084fc",
  "Skorch + sklearn":        "#e879f9",
  "Skorch Callbacks":        "#f472b6",
  "Architecture Patterns":   "#38bdf8",
  "Transfer Learning":       "#86efac",
  "DL-Specific Problems":    "#fca5a5",
  "Alternative DL Libraries":"#94a3b8",
};

const TYPE_STYLE = {
  "🔵 Concept":    { bg:"#1e3a5f", color:"#60a5fa", border:"#60a5fa" },
  "🟣 Component":  { bg:"#2a1a3a", color:"#c084fc", border:"#c084fc" },
  "🟢 Pattern":    { bg:"#1a3a2a", color:"#4ade80", border:"#4ade80" },
  "🟠 Diagnostic": { bg:"#2a1a0a", color:"#fb923c", border:"#fb923c" },
  "🔴 Problem":    { bg:"#3a0a0a", color:"#f87171", border:"#f87171" },
  "🟡 Skorch":     { bg:"#2a2a0a", color:"#facc15", border:"#facc15" },
};

const ITEMS = [
  // ═══════════════════════════════════════════
  // PYTORCH CORE
  // ═══════════════════════════════════════════
  {
    name: "Tensors & Device Management",
    lib: "torch",
    api: "torch.tensor() / .to(device) / .cuda()",
    group: "PyTorch Core",
    type: "🔵 Concept",
    description: "PyTorch's fundamental data structure. Tensors are n-dimensional arrays that can live on CPU or GPU. The .to(device) method moves them between devices. All model parameters and data must be on the same device.",
    useWhen: "Always — every PyTorch program starts here. Set device = 'cuda' if torch.cuda.is_available() else 'cpu' at the top of every script.",
    dontUse: "Don't mix devices — a tensor on CPU and a model on GPU will crash with a cryptic error. Always verify device placement before training.",
    pitfall: "Forgetting to cast input data to float32 is the #1 beginner crash. numpy arrays default to float64; PyTorch Linear layers expect float32. Always call X = X.astype(np.float32) or .float() on tensors.",
    code: `import torch
import numpy as np

# Device setup — always at the top of your script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Create tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # from Python list
x = torch.from_numpy(np.array(..., dtype=np.float32)) # from numpy (shares memory)
x = torch.zeros(32, 128)                               # zeros
x = torch.randn(32, 128)                               # random normal

# Move to device
x = x.to(device)          # preferred — works on models too
x = x.cuda()              # shorthand for CUDA

# Key dtype — nn.Linear expects float32
x = x.float()             # cast to float32
y = y.long()              # class labels must be int64 (long)

# Shape inspection
print(x.shape)            # torch.Size([32, 128])
print(x.dtype, x.device)`,
    tableLinks: "Missing Value Mishandling (dtype errors) · Feature Scale Sensitivity (scaling before tensor conversion)",
  },
  {
    name: "Autograd & Computational Graph",
    lib: "torch",
    api: "requires_grad=True / .backward() / optimizer.zero_grad()",
    group: "PyTorch Core",
    type: "🔵 Concept",
    description: "PyTorch dynamically builds a computational graph as operations are executed. Calling .backward() on the loss computes all gradients via reverse-mode automatic differentiation. This is the engine behind all neural network training.",
    useWhen: "Understanding autograd is essential for debugging training, implementing custom loss functions, and understanding why forgetting zero_grad() causes gradient accumulation bugs.",
    dontUse: "Don't call .backward() inside torch.no_grad() — gradients won't be computed. Don't access .grad before calling .backward().",
    pitfall: "The most common bug: forgetting optimizer.zero_grad() before loss.backward() causes gradients to accumulate across batches, making the effective batch size grow unboundedly and destabilising training.",
    code: `# The three lines that define every training step:
optimizer.zero_grad()   # 1. Clear accumulated gradients from previous step
loss.backward()         # 2. Compute gradients of loss w.r.t. all parameters
optimizer.step()        # 3. Update parameters using computed gradients

# Inspecting gradients (for debugging vanishing/exploding):
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.6f}")

# Disable gradient tracking (inference, validation, feature extraction):
with torch.no_grad():
    outputs = model(X_val)  # ~30% faster, lower memory

# Detach from graph (when you need a numpy value from a tensor):
loss_value = loss.detach().cpu().item()`,
    tableLinks: "Vanishing Gradients (DL Problems) · Exploding Gradients (DL Problems) · Learning Rate Issues",
  },
  {
    name: "nn.Module — Building Blocks",
    lib: "torch.nn",
    api: "class MyModel(nn.Module): def __init__ / forward",
    group: "PyTorch Core",
    type: "🔵 Concept",
    description: "The base class for all neural network modules. Subclassing nn.Module and implementing __init__ (define layers) and forward (define computation) is the standard pattern for every PyTorch model.",
    useWhen: "Every custom model definition. Layers defined as self.layer = nn.Linear(...) in __init__ are automatically registered as parameters and tracked by autograd.",
    dontUse: "Don't define layers in forward() — they won't be registered as parameters and won't be saved/loaded with model.state_dict(). Don't forget super().__init__() or parameters won't be tracked.",
    pitfall: "Defining layers inside forward() instead of __init__ means those layers are recreated on every forward pass and their parameters are never registered. This is a silent bug — the model runs but doesn't learn.",
    code: `import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.3):
        super().__init__()           # ALWAYS call this first
        # Layers defined here are registered as parameters
        self.fc1      = nn.Linear(in_features, hidden_dim)
        self.bn1      = nn.BatchNorm1d(hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.fc2      = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out      = nn.Linear(hidden_dim // 2, out_features)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)           # Raw logits — loss function applies softmax

model = MLP(20, 128, 2).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`,
    tableLinks: "MLP Classifier (sklearn table) · Architecture Patterns",
  },
  {
    name: "model.eval() vs model.train()",
    lib: "torch.nn",
    api: "model.train() / model.eval() / torch.no_grad()",
    group: "PyTorch Core",
    type: "🔵 Concept",
    description: "Two modes that change the behaviour of Dropout and BatchNorm. In train mode, dropout randomly zeroes neurons and BatchNorm uses batch statistics. In eval mode, dropout is disabled and BatchNorm uses running statistics. Forgetting to switch causes evaluation to be inconsistent.",
    useWhen: "Call model.train() at the start of each training epoch. Call model.eval() before every validation/inference step. These are NOT optional.",
    dontUse: "Never evaluate performance without model.eval() — dropout will randomly drop neurons during inference, producing different predictions each time you call the model.",
    pitfall: "Calling model.eval() without torch.no_grad() wastes memory building a gradient graph that's never used. Always use both together for validation.",
    code: `# Correct training epoch pattern:
model.train()                         # Enables dropout + BN batch stats
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    logits = model(X_batch)
    loss   = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()

# Correct validation pattern:
model.eval()                          # Disables dropout + uses BN running stats
val_losses = []
with torch.no_grad():                 # Disables gradient tracking (saves memory)
    for X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        logits     = model(X_val)
        val_loss   = criterion(logits, y_val)
        val_losses.append(val_loss.item())

print(f"Val loss: {sum(val_losses)/len(val_losses):.4f}")`,
    tableLinks: "Poor Probability Calibration (Problems) · Overfitting DL-Specific",
  },
  {
    name: "Mixed Precision Training (AMP)",
    lib: "torch.cuda.amp",
    api: "autocast() / GradScaler()",
    group: "PyTorch Core",
    type: "🔵 Concept",
    description: "Automatic Mixed Precision uses float16 for forward/backward pass computations (faster on modern GPUs with tensor cores) and float32 for parameter updates (stable). Typically provides 2-3× speedup with no accuracy loss.",
    useWhen: "Any GPU training with modern Nvidia GPUs (Volta, Turing, Ampere, Ada, Hopper). Nearly always worth enabling — it's free performance.",
    dontUse: "CPU training (AMP only speeds up CUDA tensor cores). Some custom operations don't support float16 — check for NaN losses which indicate a float16 overflow.",
    pitfall: "If loss becomes NaN during AMP training, it usually means a float16 overflow in your loss computation. Add if scaler.get_scale() < 1: break as a diagnostic. Use dynamic_scale=True (default) to handle this automatically.",
    code: `from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # Handles float16 gradient scaling to prevent underflow

model.train()
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()

    with autocast():                          # Float16 for forward pass
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)   # Loss computed in float16

    scaler.scale(loss).backward()             # Scale loss, then backprop
    scaler.unscale_(optimizer)                # Unscale before gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)                    # Update params (in float32)
    scaler.update()                           # Adjust scale factor for next step`,
    tableLinks: "Slow Training / Scalability Bottleneck (Problems) · Memory Errors / OOM",
  },

  // ═══════════════════════════════════════════
  // LAYER TYPES
  // ═══════════════════════════════════════════
  {
    name: "nn.Linear — Fully Connected",
    lib: "torch.nn",
    api: "nn.Linear(in_features, out_features, bias=True)",
    group: "Layer Types",
    type: "🟣 Component",
    description: "Applies the linear transformation y = xW^T + b. The fundamental building block of MLPs. Weights initialised with Kaiming uniform by default.",
    useWhen: "Every MLP, final classification head, projection layer in transformers, feature transformation in any architecture.",
    dontUse: "Sequential data where order matters (use LSTM/Conv1d). Image data where spatial locality matters (use Conv2d).",
    pitfall: "Output of nn.Linear has NO activation — don't forget to apply ReLU/GELU in forward(). Also: the input size must exactly match in_features or you get a dimension mismatch error.",
    code: `# Basic usage
layer = nn.Linear(128, 64)          # 128 → 64 with bias
layer = nn.Linear(128, 64, bias=False)  # no bias (common in BN layers after)

# Check weight shape
print(layer.weight.shape)           # torch.Size([64, 128])  (out, in)
print(layer.bias.shape)             # torch.Size([64])

# Fan-in initialisation (better than default for deep networks):
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)

# Typical tabular MLP block:
def mlp_block(in_f, out_f, dropout=0.3):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(dropout),
    )`,
    tableLinks: "MLP Classifier / Regressor (sklearn table) · Tabular Architecture Patterns",
  },
  {
    name: "nn.Conv1d / Conv2d — Convolutional",
    lib: "torch.nn",
    api: "nn.Conv2d(in_channels, out_channels, kernel_size) / nn.Conv1d(...)",
    group: "Layer Types",
    type: "🟣 Component",
    description: "Applies a sliding kernel over spatial (2D images) or sequential (1D signals, text) dimensions. Captures local patterns with weight sharing — far more parameter-efficient than a fully connected layer for structured inputs.",
    useWhen: "Images (Conv2d), 1D sequences / time series / NLP token sequences (Conv1d), any data with local spatial or temporal patterns.",
    dontUse: "Tabular data with no spatial/sequential structure (use Linear). Very long sequences where attention has better global receptive field.",
    pitfall: "Padding='same' keeps output size equal to input; padding='valid' reduces it. Forgetting to account for output size after convolutions + pooling causes shape mismatches at the FC layer.",
    code: `# Image classification backbone (Conv2d)
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=kernel//2)  # same padding
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 1D conv for sequential / time series (in_channels = n_features, L = sequence length)
# Input shape: (batch, channels, length)
conv1d = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, padding=2)

# Calculate output size after conv + pool (important before FC layer):
# out = (in + 2*pad - kernel) / stride + 1`,
    tableLinks: "CNN Embeddings / Transfer Learning (FE table) · Image / CV EDA",
  },
  {
    name: "nn.LSTM / nn.GRU — Recurrent",
    lib: "torch.nn",
    api: "nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)",
    group: "Layer Types",
    type: "🟣 Component",
    description: "LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) process sequences step-by-step, maintaining a hidden state. GRU is simpler (fewer parameters) and often performs comparably to LSTM. Both handle variable-length sequences.",
    useWhen: "Sequential data where order and long-range dependencies matter: time series, NLP, sensor streams. GRU is a good default if LSTM's extra parameters aren't justified by dataset size.",
    dontUse: "Long sequences > ~500 steps (transformers handle long-range dependencies better). Tabular data with no sequence structure.",
    pitfall: "LSTM returns (output, (h_n, c_n)) — both outputs and hidden state. GRU returns (output, h_n). Forgetting to unpack the tuple causes a shape error. Also: batch_first=True makes input (batch, seq, features) which is usually more convenient.",
    code: `class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,         # Input: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between layers only
            bidirectional=False,      # Set True for bidirectional (doubles hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state for classification:
        last = out[:, -1, :]          # (batch, hidden_size)
        return self.fc(self.dropout(last))`,
    tableLinks: "Time Series Features (FE table) · Lag Features · ACF/PACF",
  },
  {
    name: "nn.MultiheadAttention / Transformer",
    lib: "torch.nn",
    api: "nn.MultiheadAttention(embed_dim, num_heads) / nn.TransformerEncoder",
    group: "Layer Types",
    type: "🟣 Component",
    description: "Multi-head self-attention computes weighted attention across all positions in a sequence simultaneously, capturing global dependencies regardless of distance. The backbone of all modern NLP (BERT, GPT) and vision (ViT) models.",
    useWhen: "Long sequences where LSTM struggles with long-range dependencies. NLP tasks with transformer models. Tabular data with FT-Transformer. Any task where you have a GPU and enough data.",
    dontUse: "Very short sequences (LSTM is often better). Very small datasets (transformers need more data than LSTMs). Edge devices with strict memory constraints.",
    pitfall: "TransformerEncoder expects src_key_padding_mask (True = position to IGNORE) which is the opposite of most intuitions. Incorrectly setting this mask is a common silent bug that makes the model attend to padding tokens.",
    code: `# Minimal Transformer encoder for sequence classification
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len=512):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed  = nn.Embedding(max_len, embed_dim)   # learned positional
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc          = nn.Linear(embed_dim, num_classes)

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len) — token IDs
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.fc(x[:, 0, :])   # CLS token`,
    tableLinks: "Contextual Sentence Embeddings (FE table) · TF-IDF + SVD (FE table)",
  },
  {
    name: "nn.Embedding",
    lib: "torch.nn",
    api: "nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)",
    group: "Layer Types",
    type: "🟣 Component",
    description: "Lookup table that maps discrete integer IDs (words, categories, users) to dense vectors. Learns the embeddings during training or can be initialised with pre-trained vectors (GloVe, FastText).",
    useWhen: "Any discrete input: token IDs in NLP, category IDs in tabular data, user/item IDs in recommendation systems. An alternative to OHE that learns a dense semantic representation.",
    dontUse: "Low-cardinality categories where OHE suffices and the embedding wouldn't be meaningfully learned with limited data.",
    pitfall: "Embedding dimensions should follow a rule of thumb: min(50, (cardinality + 1) // 2). Too large and the embedding overfits; too small and it under-represents the category space.",
    code: `# Word embedding lookup:
embedding = nn.Embedding(vocab_size=10000, embedding_dim=128, padding_idx=0)
# token_ids: (batch, seq_len) of int64
embedded = embedding(token_ids)   # → (batch, seq_len, 128)

# Load pre-trained GloVe weights:
# pretrained_weight = load_glove_matrix(...)  # shape: (vocab_size, embedding_dim)
# embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
# embedding.weight.requires_grad = False  # freeze for first few epochs

# Categorical embedding for tabular (entity embedding):
cat_embedding = nn.Embedding(num_categories, embedding_dim=8)
embedded_cat  = cat_embedding(category_ids).squeeze(1)  # (batch, 8)`,
    tableLinks: "One-Hot Encoding (FE table) · Word Embeddings (FE table) · CatBoost Encoding",
  },
  {
    name: "BatchNorm / LayerNorm / GroupNorm",
    lib: "torch.nn",
    api: "nn.BatchNorm1d(features) / nn.LayerNorm(features)",
    group: "Layer Types",
    type: "🟣 Component",
    description: "Normalisation layers that stabilise training by standardising activations. BatchNorm normalises over the batch dimension (fast, but batch-size dependent). LayerNorm normalises over the feature dimension (batch-size independent, preferred in Transformers). GroupNorm is between the two.",
    useWhen: "BatchNorm: CNNs, MLPs with large batches. LayerNorm: Transformers, RNNs, NLP, small batch sizes. GroupNorm: detection/segmentation models with small batches.",
    dontUse: "BatchNorm with batch size < 8 — statistics become unreliable. BatchNorm in RNNs (use LayerNorm instead). Any normalisation when model.eval() mode behaviour must exactly match single-sample inference.",
    pitfall: "BatchNorm behaves differently in train() vs eval() mode. If you call model.eval() and use batch statistics, predictions will be inconsistent. Always use running statistics (eval mode) for inference.",
    code: `# BatchNorm in MLP (after Linear, before activation)
nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),   # Normalise over batch, per feature
    nn.ReLU(),
    nn.Dropout(0.3),
)

# LayerNorm in Transformer feed-forward (after residual addition)
nn.Sequential(
    nn.Linear(128, 64),
    nn.LayerNorm(64),     # Normalise over feature dim, per sample
    nn.GELU(),
)

# Inspect running statistics (useful for debugging eval mode discrepancy):
bn = nn.BatchNorm1d(64)
print(bn.running_mean[:5])   # What BatchNorm uses during eval()
print(bn.running_var[:5])`,
    tableLinks: "Vanishing Gradients (DL Problems) · Overfitting DL-Specific · Batch Size Effects (Problems)",
  },

  // ═══════════════════════════════════════════
  // ACTIVATION FUNCTIONS
  // ═══════════════════════════════════════════
  {
    name: "ReLU / LeakyReLU / ELU",
    lib: "torch.nn / torch.nn.functional",
    api: "nn.ReLU() / nn.LeakyReLU(0.01) / F.relu(x)",
    group: "Activation Functions",
    type: "🟣 Component",
    description: "ReLU (Rectified Linear Unit) = max(0, x). The default activation for most MLPs and CNNs. LeakyReLU passes a small gradient for negative inputs (avoids dying ReLU). ELU has smooth negative outputs that push mean activations toward zero.",
    useWhen: "ReLU: CNNs, most MLPs, default choice when unsure. LeakyReLU or ELU: when you observe dying ReLU (neurons always outputting 0). ELU often slightly outperforms ReLU on deep networks.",
    dontUse: "ReLU in the final output layer (use Sigmoid for binary, Softmax for multi-class, linear for regression). ReLU in very deep residual networks where GELU performs better.",
    pitfall: "nn.ReLU(inplace=True) modifies the input tensor in-place — faster but can cause errors if the same tensor is needed for gradient computation during backward pass. Use inplace=False when debugging.",
    code: `# Comparison of activations
import torch
x = torch.randn(5)

print("ReLU:     ", torch.relu(x))          # max(0, x)
print("LeakyReLU:", F.leaky_relu(x, 0.01)) # x if x>0 else 0.01*x
print("ELU:      ", F.elu(x))              # x if x>0 else α*(exp(x)-1)
print("GELU:     ", F.gelu(x))             # x * Φ(x), smooth approximation

# In model definition:
self.act = nn.ReLU()                       # Creates a module (recommended in __init__)
x = F.relu(x)                             # Functional (no state, fine in forward())`,
    tableLinks: "Vanishing Gradients (DL Problems) · Dead Neurons (DL Problems)",
  },
  {
    name: "GELU / Swish / Mish",
    lib: "torch.nn / torch.nn.functional",
    api: "nn.GELU() / F.silu(x) [Swish]",
    group: "Activation Functions",
    type: "🟣 Component",
    description: "Modern smooth activation functions that outperform ReLU on language models and vision transformers. GELU (used in BERT, GPT) is Gaussian-gated. Swish/SiLU (used in EfficientNet) is x·σ(x). Both avoid the dying neuron problem of ReLU.",
    useWhen: "Transformer architectures (GELU is standard). EfficientNet/MobileNetV3 (Swish). Any MLP where you want to try beating ReLU — often 0.5-1% accuracy improvement.",
    dontUse: "Embedded/edge inference where the exp() computation is too slow. Older architectures that were specifically tuned for ReLU.",
    pitfall: "GELU and Swish are slower than ReLU due to the gaussian/sigmoid computation. On very large models, this adds up. Use ReLU for production latency-critical models unless the accuracy difference justifies it.",
    code: `# GELU — default in transformer architectures
nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),              # Smooth, differentiable everywhere
    nn.Linear(2048, 512),
)

# Swish / SiLU — default in EfficientNet family
self.act = nn.SiLU()       # F.silu(x) = x * sigmoid(x)

# Custom Mish activation (if needed):
def mish(x):
    return x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + e^x))`,
    tableLinks: "Architecture Patterns · Transfer Learning",
  },
  {
    name: "Sigmoid / Tanh",
    lib: "torch.nn",
    api: "nn.Sigmoid() / nn.Tanh() / torch.sigmoid(x)",
    group: "Activation Functions",
    type: "🟣 Component",
    description: "Sigmoid squashes output to (0,1) — used as the final layer for binary classification. Tanh squashes to (-1,1) — used in LSTM gates. Both saturate for large |x|, causing vanishing gradients in intermediate layers.",
    useWhen: "Sigmoid: final layer for binary classification (output = probability). Tanh: LSTM/GRU gate activations (handled automatically by nn.LSTM). Tanh for image pixel generation outputs.",
    dontUse: "Hidden layers of deep networks — saturation kills gradients. Use ReLU/GELU instead for hidden layers.",
    pitfall: "Using nn.Sigmoid() in the final layer and then passing outputs to nn.BCEWithLogitsLoss() — BCEWithLogitsLoss applies sigmoid internally. Double-sigmoid makes predictions numerically unstable. Use raw logits with BCEWithLogitsLoss.",
    code: `# WRONG: double sigmoid — don't do this
output = torch.sigmoid(self.fc(x))        # sigmoid in forward()
loss = nn.BCEWithLogitsLoss()(output, y)  # applies sigmoid AGAIN → wrong!

# CORRECT: logits → BCEWithLogitsLoss (numerically stable)
logits = self.fc(x)                         # raw output, no activation
loss   = nn.BCEWithLogitsLoss()(logits, y)  # internally stable sigmoid

# For inference / probability output:
proba = torch.sigmoid(logits)               # apply sigmoid only for final proba`,
    tableLinks: "Loss Functions (BCEWithLogits) · Poor Probability Calibration",
  },
  {
    name: "Softmax / Log-Softmax",
    lib: "torch.nn",
    api: "nn.Softmax(dim=1) / F.log_softmax(x, dim=1)",
    group: "Activation Functions",
    type: "🟣 Component",
    description: "Softmax converts a vector of logits into a probability distribution summing to 1. Log-softmax is the numerically stable version used with NLLLoss. For multi-class classification, CrossEntropyLoss combines log-softmax + NLLLoss internally.",
    useWhen: "Final layer for multi-class output — but only for inference/probability display. Never apply softmax before CrossEntropyLoss.",
    dontUse: "Before nn.CrossEntropyLoss — it applies log-softmax internally. Applying softmax before CrossEntropyLoss causes log(softmax(logits)) = double-softmax and produces incorrect gradients.",
    pitfall: "dim argument is critical — softmax(dim=1) for (batch, classes) shaped output. Wrong dim causes softmax over the wrong axis and numerically nonsensical results.",
    code: `# WRONG: softmax before CrossEntropyLoss
logits = self.fc(x)
proba  = F.softmax(logits, dim=1)   # Don't do this
loss   = nn.CrossEntropyLoss()(proba, y)  # CrossEntropyLoss reapplies log_softmax!

# CORRECT: raw logits to CrossEntropyLoss
logits = self.fc(x)                  # Raw unnormalised scores
loss   = nn.CrossEntropyLoss()(logits, y)  # Handles softmax internally

# For inference only — convert to probabilities:
with torch.no_grad():
    proba = F.softmax(model(X), dim=1)          # Class probabilities
    pred  = torch.argmax(proba, dim=1)          # Predicted class`,
    tableLinks: "Loss Functions (CrossEntropyLoss) · Threshold Set at Default 0.5",
  },

  // ═══════════════════════════════════════════
  // LOSS FUNCTIONS
  // ═══════════════════════════════════════════
  {
    name: "nn.CrossEntropyLoss",
    lib: "torch.nn",
    api: "nn.CrossEntropyLoss(weight=None, label_smoothing=0.0)",
    group: "Loss Functions",
    type: "🟣 Component",
    description: "Combines log-softmax and negative log-likelihood in one numerically stable operation. The standard loss for multi-class classification. Accepts raw logits (no softmax needed). Supports class weighting for imbalance and label smoothing.",
    useWhen: "Multi-class classification (C > 2). Default choice when you're unsure which classification loss to use.",
    dontUse: "Binary classification with single output node (use BCEWithLogitsLoss). Regression (use MSELoss/HuberLoss). Multi-label classification (use BCEWithLogitsLoss on each label independently).",
    pitfall: "Target y must be integer class indices (torch.long / int64), NOT one-hot vectors. Passing a float one-hot vector causes a cryptic dimension mismatch error.",
    code: `criterion = nn.CrossEntropyLoss()                    # Balanced classes
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing (reduces overconfidence)

# Class weighting for imbalanced data (analogous to class_weight='balanced' in sklearn):
counts  = torch.bincount(y_train)
weights = 1.0 / counts.float()
weights = weights / weights.sum()                     # Normalise
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Usage:
# logits: (batch, num_classes) — raw outputs, no softmax
# y:      (batch,)             — integer class indices, dtype=torch.long
loss = criterion(logits, y)   # logits is (N, C), y is (N,)`,
    tableLinks: "Class Imbalance (Problems) · Label / Annotation Noise (Problems) · Label Smoothing",
  },
  {
    name: "nn.BCEWithLogitsLoss",
    lib: "torch.nn",
    api: "nn.BCEWithLogitsLoss(pos_weight=None)",
    group: "Loss Functions",
    type: "🟣 Component",
    description: "Binary Cross-Entropy with built-in sigmoid — numerically stable combination. For binary classification (single output neuron) or multi-label classification (multiple independent binary predictions). The pos_weight parameter handles class imbalance.",
    useWhen: "Binary classification (model output is a single logit). Multi-label classification where each label is independent (pass a vector of logits and a vector of 0/1 targets).",
    dontUse: "Multi-class (mutually exclusive) classification — use CrossEntropyLoss. When model already applies sigmoid (use plain BCELoss instead, though BCEWithLogitsLoss is preferred).",
    pitfall: "pos_weight must be a tensor, not a scalar. For 10:1 imbalance (10 negatives per positive), set pos_weight=torch.tensor([10.0]) to up-weight the positive class.",
    code: `# Binary classification
criterion = nn.BCEWithLogitsLoss()

# With positive class weighting for imbalance:
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
pos_weight   = torch.tensor([n_neg / n_pos]).to(device)
criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Model output: single logit per sample (no sigmoid in forward())
# y target: float tensor of 0.0 or 1.0
logit = self.fc(x)                          # Shape: (batch, 1) or (batch,)
loss  = criterion(logit.squeeze(), y.float())

# For multi-label (e.g., 10 independent binary labels):
# logits: (batch, 10), targets: (batch, 10) of 0/1 floats
loss = criterion(multi_logits, multi_targets.float())`,
    tableLinks: "Class Imbalance (Problems) · Threshold Set at Default 0.5 (Problems)",
  },
  {
    name: "nn.MSELoss / nn.L1Loss / nn.HuberLoss",
    lib: "torch.nn",
    api: "nn.MSELoss() / nn.L1Loss() / nn.HuberLoss(delta=1.0)",
    group: "Loss Functions",
    type: "🟣 Component",
    description: "Regression losses. MSE (Mean Squared Error) penalises large errors quadratically — sensitive to outliers. L1 (Mean Absolute Error) is robust to outliers but non-differentiable at zero. HuberLoss is quadratic near zero (like MSE) and linear for large errors (like L1) — the best of both.",
    useWhen: "MSE: clean data, symmetric errors, when large errors truly are worse. L1: robust regression, sparse-target scenarios. HuberLoss: most real-world regression with potential outliers — this should be your default regression loss.",
    dontUse: "MSE when your data has significant outliers (use HuberLoss). L1 when you need smooth gradients everywhere (use HuberLoss).",
    pitfall: "MSE and L1 return reduction='mean' by default — the loss is averaged over all samples. If sample weights are needed, set reduction='none' and manually multiply by weights before taking the mean.",
    code: `criterion_mse   = nn.MSELoss()                   # L2 loss
criterion_mae   = nn.L1Loss()                    # L1 loss (robust)
criterion_huber = nn.HuberLoss(delta=1.0)        # Smooth L1/Huber (best default)

# Log-scale regression trick (for right-skewed targets like price/sales):
# 1. log-transform target: y_log = torch.log1p(y)
# 2. predict y_log with MSELoss
# 3. exponentiate predictions: pred = torch.expm1(model(X))
# This equivalent to training with RMSLE on original scale

# Custom weighted regression:
criterion_none = nn.MSELoss(reduction='none')   # Returns per-sample loss
per_sample_loss = criterion_none(preds, targets) # (batch,)
weighted_loss   = (per_sample_loss * sample_weights).mean()`,
    tableLinks: "Log Transform (FE table) · Wrong Evaluation Metric (Problems) · Metric Mismatch (Problems)",
  },
  {
    name: "Focal Loss",
    lib: "custom (torchvision has FocalLoss)",
    api: "FocalLoss(alpha=0.25, gamma=2.0) — manual or torchvision.ops.sigmoid_focal_loss",
    group: "Loss Functions",
    type: "🟣 Component",
    description: "Extension of BCEWithLogitsLoss that down-weights easy (well-classified) examples and focuses training on hard, misclassified ones. The gamma parameter controls the focusing strength. Originally developed for object detection with extreme imbalance.",
    useWhen: "Extreme class imbalance (> 100:1). Object detection. Any task where most training examples are easy negatives and the hard positives drive the task difficulty.",
    dontUse: "Moderate imbalance (class_weight or pos_weight handles this fine). Multi-class classification (focal loss is primarily for binary/detection).",
    pitfall: "gamma=2 is the default from the original paper but may not be optimal for your task. Treat it as a hyperparameter and tune in [0.5, 1, 2, 5].",
    code: `# Manual focal loss implementation (clear and easy to use):
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    proba    = torch.sigmoid(logits)
    pt       = torch.where(targets == 1, proba, 1 - proba)  # p_t
    focal_w  = alpha * (1 - pt) ** gamma
    return (focal_w * bce_loss).mean()

# Or use torchvision (installed with PyTorch):
from torchvision.ops import sigmoid_focal_loss
loss = sigmoid_focal_loss(logits, targets.float(), alpha=0.25, gamma=2.0, reduction='mean')`,
    tableLinks: "Class Imbalance (Problems) · SMOTE Applied Before Split (Problems)",
  },
  {
    name: "Label Smoothing",
    lib: "torch.nn",
    api: "nn.CrossEntropyLoss(label_smoothing=0.1)",
    group: "Loss Functions",
    type: "🟣 Component",
    description: "Replaces hard 0/1 labels with soft targets (e.g., 0.05 and 0.95). Prevents overconfident predictions, acts as regularisation, and improves calibration. Especially effective when annotation noise is present.",
    useWhen: "Any multi-class classification with moderate-to-large datasets. When model outputs overly confident probabilities. Label noise in annotations.",
    dontUse: "Very small datasets (soft labels add noise that hurts learning when data is already scarce). Knowledge distillation (it conflicts with the soft teacher labels).",
    pitfall: "Label smoothing typically uses ε=0.1 as a safe default. Values > 0.2 hurt performance because the model can no longer learn strong class separation.",
    code: `# Built-in in PyTorch >= 1.10 (simplest approach):
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Internally: soft_target = (1-ε)*one_hot + ε/C

# Manual soft labels (for custom control):
def smooth_labels(y, n_classes, eps=0.1):
    one_hot = F.one_hot(y, n_classes).float()
    return one_hot * (1 - eps) + eps / n_classes

# In skorch, pass criterion__label_smoothing=0.1:
net = NeuralNetClassifier(
    module=MyModel,
    criterion=nn.CrossEntropyLoss,
    criterion__label_smoothing=0.1,
)`,
    tableLinks: "Label / Annotation Noise (Problems) · Poor Probability Calibration (Problems)",
  },

  // ═══════════════════════════════════════════
  // OPTIMIZERS
  // ═══════════════════════════════════════════
  {
    name: "Adam",
    lib: "torch.optim",
    api: "torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8)",
    group: "Optimizers",
    type: "🟣 Component",
    description: "Adaptive Moment Estimation. Maintains per-parameter learning rates adapted from first and second moment estimates of gradients. Converges fast and is robust to learning rate choice. The most widely used DL optimiser.",
    useWhen: "Default choice for most deep learning tasks. Good starting point for any new architecture.",
    dontUse: "When AdamW outperforms (weight decay implementation in Adam is incorrect — AdamW should be preferred for regularised models). Some tasks (image classification) where SGD with momentum achieves better generalisation with the right LR schedule.",
    pitfall: "Adam's built-in weight_decay (L2 regularisation) is applied to the adaptive updates, not to the weights directly — this is mathematically incorrect. Use AdamW for correct weight decay.",
    code: `optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,          # Most important hyperparameter — tune first
    betas=(0.9, 0.999),  # Momentum for grad mean and grad² mean
    eps=1e-8,         # Numerical stability (increase to 1e-7 if NaN losses)
    weight_decay=0.0  # Use AdamW instead if you want correct weight decay
)

# Inspect current learning rate (after scheduler steps):
current_lr = optimizer.param_groups[0]['lr']
print(f"Current LR: {current_lr:.2e}")`,
    tableLinks: "Learning Rate Too High / Too Low (Problems) · Hyperparameter Tuning (Problems)",
  },
  {
    name: "AdamW",
    lib: "torch.optim",
    api: "torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)",
    group: "Optimizers",
    type: "🟣 Component",
    description: "Adam with corrected weight decay. Decouples the weight decay from the adaptive learning rate, which is mathematically the correct way to apply L2 regularisation. Standard in modern transformer training (BERT, GPT, ViT).",
    useWhen: "Default for transformer models, large models, any task where regularisation matters. Should generally be preferred over Adam when using weight_decay.",
    dontUse: "When weight decay is not needed (Adam and AdamW are equivalent). On small networks with few parameters where regularisation has no benefit.",
    pitfall: "Default weight_decay=0.01 is often too weak. For transformers, 0.1 is common. For small MLPs, 1e-4 is a good start. Always tune weight_decay.",
    code: `optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,             # Lower than Adam default for transformers
    weight_decay=0.01,   # L2 regularisation strength — tune this
    betas=(0.9, 0.999),
)

# For fine-tuning transformers: different LR for different layers
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},   # Pretrained — low LR
    {'params': model.head.parameters(),     'lr': 1e-3},   # New head — high LR
], weight_decay=0.01)`,
    tableLinks: "Overfitting — High Variance (Problems) · Transfer Learning",
  },
  {
    name: "SGD with Momentum",
    lib: "torch.optim",
    api: "torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)",
    group: "Optimizers",
    type: "🟣 Component",
    description: "Classical stochastic gradient descent with momentum accumulation. Harder to tune than Adam but can find better generalisable solutions (flatter minima) with the right learning rate schedule. Standard for ImageNet-scale image classification.",
    useWhen: "Computer vision (ResNet, VGG training on ImageNet). When training from scratch with a cosine/step decay schedule. Research that needs to match published SGD-based results.",
    dontUse: "NLP/transformer training (Adam/AdamW is standard). Quick prototyping where Adam's robustness is valuable. Fine-tuning pretrained models (Adam/AdamW converges faster).",
    pitfall: "SGD is very sensitive to learning rate choice — 10× too high causes divergence; 10× too low causes extremely slow convergence. Must be paired with a LR schedule (OneCycleLR or cosine annealing).",
    code: `optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,            # High initial LR — schedule will decay it
    momentum=0.9,      # Accumulate gradient direction
    weight_decay=1e-4, # L2 regularisation
    nesterov=True,     # Lookahead version — usually slightly better
)
# Always pair with a learning rate schedule when using SGD:
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=100
)`,
    tableLinks: "LR Schedulers · Overfitting — High Variance",
  },

  // ═══════════════════════════════════════════
  // LR SCHEDULERS
  // ═══════════════════════════════════════════
  {
    name: "ReduceLROnPlateau",
    lib: "torch.optim.lr_scheduler",
    api: "ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)",
    group: "LR Schedulers",
    type: "🟣 Component",
    description: "Reduces learning rate by a factor when a monitored metric stops improving for 'patience' epochs. Adaptive — reacts to actual training progress rather than a fixed schedule. The safest default scheduler.",
    useWhen: "When you don't know how many epochs to train. When you want set-and-forget scheduling. General-purpose training where convergence speed varies.",
    dontUse: "When training for a fixed number of epochs (OneCycleLR or CosineAnnealing are better). Transformer fine-tuning (linear warmup + cosine decay is standard).",
    pitfall: "Must be stepped with the validation metric, not the training loss. scheduler.step(val_loss) — not scheduler.step() — or it won't trigger correctly.",
    code: `scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 'min' for loss, 'max' for accuracy/AUC
    factor=0.5,        # LR multiplied by this factor on plateau
    patience=5,        # Number of epochs with no improvement before reducing
    min_lr=1e-7,       # Lower bound on LR
    verbose=True,      # Print LR changes
)

# In training loop — call AFTER computing validation loss:
scheduler.step(val_loss)  # NOT scheduler.step() alone`,
    tableLinks: "Early Stopping Misconfiguration (Problems) · Learning Rate Too High / Too Low",
  },
  {
    name: "OneCycleLR",
    lib: "torch.optim.lr_scheduler",
    api: "OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=N, epochs=E)",
    group: "LR Schedulers",
    type: "🟣 Component",
    description: "Cyclic LR schedule with warmup → peak → decay in one training run. Starts at max_lr/div_factor, rises to max_lr, then decays to max_lr/(div_factor * final_div_factor). Developed by fastai — often trains to better accuracy in fewer epochs than fixed schedules.",
    useWhen: "CNN training. When you want to train fast and achieve good results without extensive tuning. Computer vision tasks with SGD. The '1cycle policy' is a well-established high-performance schedule.",
    dontUse: "When training continues past the cycle (OneCycleLR is designed for exactly one cycle). Fine-tuning with tiny LR variations.",
    pitfall: "OneCycleLR is called per step, not per epoch — scheduler.step() inside the batch loop, not after each epoch. Calling it per epoch causes the LR to advance far too slowly.",
    code: `optimizer  = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler  = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,                         # Peak learning rate
    steps_per_epoch=len(train_loader),   # Number of batches per epoch
    epochs=30,                           # Total epochs
    pct_start=0.3,                       # 30% warmup, 70% decay
)

for epoch in range(30):
    model.train()
    for X_batch, y_batch in train_loader:
        # ... forward + backward + optimizer.step() ...
        scheduler.step()    # ← called per BATCH, not per epoch`,
    tableLinks: "Learning Rate Too High / Too Low (Problems) · Slow Training / Scalability",
  },
  {
    name: "CosineAnnealingLR / CosineAnnealingWarmRestarts",
    lib: "torch.optim.lr_scheduler",
    api: "CosineAnnealingLR(optimizer, T_max=100) / CosineAnnealingWarmRestarts(optimizer, T_0=10)",
    group: "LR Schedulers",
    type: "🟣 Component",
    description: "CosineAnnealingLR decays LR from max to min following a cosine curve over T_max epochs. WarmRestarts periodically resets LR, allowing the model to escape local minima. Both are standard for transformer fine-tuning and modern CV training.",
    useWhen: "Transformer fine-tuning (cosine decay after linear warmup). Long training runs where you want smooth LR decay. Ensemble diversity — different restart cycles give different local minima for ensembling.",
    dontUse: "Short training runs of < 10 epochs. When OneCycleLR or ReduceLROnPlateau serve your use case better.",
    pitfall: "CosineAnnealingLR needs T_max = total training epochs. Setting T_max incorrectly means LR doesn't hit its minimum at the end of training.",
    code: `# Cosine annealing — standard for transformer fine-tuning
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,       # Total epochs for one cosine cycle
    eta_min=1e-7,    # Minimum LR at end of cycle
)

# Warm restarts — for ensembling via snapshot ensembles
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # Initial restart period (epochs)
    T_mult=2,        # Each restart takes 2× longer
    eta_min=1e-7,
)

# Linear warmup + cosine decay (standard transformer recipe):
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=100)
cosine  = CosineAnnealingLR(optimizer, T_max=900, eta_min=1e-7)
sched   = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[100])`,
    tableLinks: "Transfer Learning · Ensemble Diversity Collapse (Problems)",
  },

  // ═══════════════════════════════════════════
  // REGULARISATION
  // ═══════════════════════════════════════════
  {
    name: "Dropout",
    lib: "torch.nn",
    api: "nn.Dropout(p=0.5) / nn.Dropout2d(p=0.2)",
    group: "Regularisation",
    type: "🟣 Component",
    description: "Randomly zeroes a fraction p of neuron activations during training. Forces the network to learn redundant representations, acting as a powerful regulariser. Disabled automatically in eval() mode.",
    useWhen: "Any MLP or dense layer with risk of overfitting. p=0.1-0.3 for smaller models, p=0.4-0.5 for large MLPs. Dropout2d for convolutional feature maps.",
    dontUse: "Small networks where underfitting is the problem. Immediately after BatchNorm (they can conflict — use one or the other for best results). Before the output layer in small networks.",
    pitfall: "If model.eval() is never called during validation, dropout remains active and produces noisy, non-deterministic validation predictions — making CV scores unreliable.",
    code: `class RegularisedMLP(nn.Module):
    def __init__(self, in_f, hidden, out_f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),          # After activation, before next layer
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),          # Less dropout deeper in the network
            nn.Linear(hidden // 2, out_f),
        )

    def forward(self, x):
        return self.net(x)

# Check: dropout active in train, inactive in eval:
model.train(); print(model(x))   # Stochastic
model.eval();  print(model(x))   # Deterministic`,
    tableLinks: "Overfitting — High Variance (Problems) · Overfitting DL-Specific",
  },
  {
    name: "Weight Decay (L2 in Optimizer)",
    lib: "torch.optim",
    api: "optimizer = AdamW(model.parameters(), weight_decay=0.01)",
    group: "Regularisation",
    type: "🟣 Component",
    description: "Adds λ||W||² to the loss, penalising large weights. In AdamW, this is correctly decoupled from the adaptive LR update. Equivalent to L2 regularisation in sklearn. The most fundamental neural network regulariser.",
    useWhen: "Always use some weight decay — it's essentially free regularisation. 1e-4 for small networks, 1e-2 for transformers, 0.1 for large language models.",
    dontUse: "Usually don't apply weight decay to biases and BatchNorm parameters — they don't benefit from L2 regularisation.",
    pitfall: "Applying weight decay to all parameters including LayerNorm scale/bias and embedding layers can hurt performance. Separate parameter groups for weight-decayed vs non-decayed parameters.",
    code: `# Best practice: separate weight decay from biases and norm params:
decay_params    = [p for n, p in model.named_parameters()
                   if p.ndim >= 2 and 'norm' not in n]   # Weights (matrices)
no_decay_params = [p for n, p in model.named_parameters()
                   if p.ndim < 2 or 'norm' in n]         # Biases, LayerNorm

optimizer = torch.optim.AdamW([
    {'params': decay_params,    'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},   # No decay for biases
], lr=1e-4)`,
    tableLinks: "Overfitting — High Variance (Problems) · Lasso / Ridge (sklearn table)",
  },
  {
    name: "Gradient Clipping",
    lib: "torch.nn.utils",
    api: "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    group: "Regularisation",
    type: "🟣 Component",
    description: "Rescales the gradient vector so its norm never exceeds max_norm. The single most reliable fix for exploding gradients. Standard in all transformer and RNN training.",
    useWhen: "Always for RNN/LSTM/GRU training. Always for transformer training. Any time you see NaN loss or wildly oscillating loss curves.",
    dontUse: "Very small max_norm (< 0.1) that would clip too aggressively and slow convergence. If you're using gradient clipping as a band-aid for a learning rate that's simply too high, fix the LR instead.",
    pitfall: "Gradient clipping must happen AFTER loss.backward() and BEFORE optimizer.step(). With AMP (mixed precision), unscale gradients first before clipping: scaler.unscale_(optimizer).",
    code: `# Standard training step with gradient clipping:
optimizer.zero_grad()
loss.backward()

# For AMP training, unscale first:
# scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# clip_grad_value_ clips each gradient individually (less common):
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

optimizer.step()

# Monitor gradient norms to choose max_norm:
total_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"Grad norm before clip: {total_norm:.4f}")`,
    tableLinks: "Exploding Gradients (DL Problems) · Mixed Precision Training",
  },
  {
    name: "Mixup Augmentation",
    lib: "torch (manual) / timm",
    api: "x_mix = λ*x1 + (1-λ)*x2; y_mix = λ*y1 + (1-λ)*y2",
    group: "Regularisation",
    type: "🟢 Pattern",
    description: "Creates virtual training samples by linearly interpolating between pairs of examples (and their labels). Encourages the model to behave linearly between training examples, acting as a strong regulariser. Especially effective for image classification and NLP.",
    useWhen: "Any classification task with large enough datasets (> 10k samples). Image classification — where it consistently improves by 0.5-2%. Competition settings where regularisation is paramount.",
    dontUse: "Regression tasks (mixed labels lose meaning). Very small datasets (< 1k samples) where the interpolated samples are noisy. Object detection (mixed bounding boxes are hard to define).",
    pitfall: "Mixup requires computing loss against soft labels, not hard integer labels. Use CrossEntropyLoss with soft targets or manually compute label-weighted loss.",
    code: `import numpy as np

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size  = x.size(0)
    index       = torch.randperm(batch_size, device=x.device)
    mixed_x     = lam * x + (1 - lam) * x[index]
    y_a, y_b    = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop:
X_mix, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=0.2)
logits   = model(X_mix)
loss     = mixup_criterion(criterion, logits, y_a, y_b, lam)`,
    tableLinks: "Overfitting — High Variance (Problems) · Data Augmentation (FE table)",
  },

  // ═══════════════════════════════════════════
  // DATA PIPELINE
  // ═══════════════════════════════════════════
  {
    name: "Custom Dataset (torch.utils.data.Dataset)",
    lib: "torch.utils.data",
    api: "class MyDataset(Dataset): __len__ / __getitem__",
    group: "Data Pipeline",
    type: "🟢 Pattern",
    description: "The standard way to wrap any data source for PyTorch. Implementing __len__ (total size) and __getitem__ (single sample retrieval) makes your data compatible with DataLoader, enabling batching, shuffling, and parallel loading automatically.",
    useWhen: "Any custom data source: pandas DataFrames, numpy arrays, image files on disk, HDF5 files, databases. Always wrap your data in a Dataset before passing to DataLoader.",
    dontUse: "Simple numpy arrays passed directly — while technically possible with TensorDataset, a custom Dataset gives you transform flexibility and lazy loading.",
    pitfall: "Loading all data in __init__ causes OOM for large datasets. Use lazy loading in __getitem__ — only load the specific sample when requested. This is the key to handling image datasets that don't fit in RAM.",
    code: `from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        # Store as tensors immediately
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Lazy image dataset (load from disk on demand):
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')  # Load on demand
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]`,
    tableLinks: "Data Augmentation (FE table) · Memory Errors / OOM (Problems)",
  },
  {
    name: "DataLoader",
    lib: "torch.utils.data",
    api: "DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)",
    group: "Data Pipeline",
    type: "🟢 Pattern",
    description: "Wraps a Dataset and provides batched iteration with optional shuffling, multi-process data loading (num_workers), and memory pinning (pin_memory) for faster GPU transfer. The interface between your data and your training loop.",
    useWhen: "Every training pipeline. Always set num_workers > 0 for image datasets to avoid CPU bottleneck. pin_memory=True when training on GPU.",
    dontUse: "num_workers > 0 on Windows without a `if __name__ == '__main__'` guard — causes multiprocessing errors on Windows.",
    pitfall: "Setting num_workers too high on a shared machine or small dataset adds overhead from spawning processes that exceeds the loading speedup. Start with num_workers=4 and profile.",
    code: `from torch.utils.data import DataLoader

train_ds = TabularDataset(X_train, y_train)
val_ds   = TabularDataset(X_val,   y_val)

train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,       # Shuffle training data every epoch
    num_workers=4,      # Parallel loading — 0 for debugging, 4 for production
    pin_memory=True,    # Faster RAM → GPU transfer (only with CUDA)
    drop_last=True,     # Drop last incomplete batch (avoids BatchNorm issues with batch_size=1)
)
val_loader = DataLoader(
    val_ds,
    batch_size=512,     # Can use larger batch for validation (no gradients)
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)`,
    tableLinks: "Slow Training / Scalability Bottleneck (Problems) · Missing Value Mishandling",
  },
  {
    name: "WeightedRandomSampler",
    lib: "torch.utils.data",
    api: "WeightedRandomSampler(weights, num_samples, replacement=True)",
    group: "Data Pipeline",
    type: "🟢 Pattern",
    description: "Over-samples minority class examples at the DataLoader level by assigning a sampling weight to each training sample. An alternative to SMOTE that doesn't create synthetic samples — just samples existing ones more frequently.",
    useWhen: "Imbalanced classification — an alternative to class_weight in the loss function. More flexible: can implement curriculum learning or hard-example mining by adjusting weights dynamically.",
    dontUse: "When class_weight in the loss function is sufficient (simpler). When using batch normalisation with very imbalanced weights (majority class under-sampling can cause instability).",
    pitfall: "Weights must be per-sample, not per-class. Compute per-sample weight as class_weight[class_of_sample] for each sample in the training set.",
    code: `from torch.utils.data import WeightedRandomSampler

# Compute per-sample weights:
y_train_np  = np.array(y_train)
class_counts = np.bincount(y_train_np)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train_np]   # Weight for each training sample

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(sample_weights),
    replacement=True,   # Sample with replacement (some samples appear multiple times)
)

# Use sampler instead of shuffle=True:
train_loader = DataLoader(
    train_ds,
    batch_size=256,
    sampler=sampler,    # NOTE: shuffle must be False when using sampler
    num_workers=4,
)`,
    tableLinks: "Class Imbalance (Problems) · SMOTE Applied Before Split (Problems)",
  },

  // ═══════════════════════════════════════════
  // TRAINING PATTERNS
  // ═══════════════════════════════════════════
  {
    name: "The Complete Training Loop",
    lib: "torch",
    api: "model.train() → zero_grad → forward → loss → backward → clip → step",
    group: "Training Patterns",
    type: "🟢 Pattern",
    description: "The canonical PyTorch training loop structure. Every DL training job follows this exact pattern: iterate over batches, zero gradients, forward pass, compute loss, backward pass, clip gradients, update parameters.",
    useWhen: "Every custom PyTorch training job. Understanding this loop is the foundation — Skorch, Lightning, and Trainer abstractions all implement this pattern internally.",
    dontUse: "No reason to avoid — this is fundamental. Use Skorch or Lightning to avoid writing this boilerplate, but understand what they're doing under the hood.",
    pitfall: "The five deadly omissions: (1) not calling zero_grad, (2) not calling backward, (3) not calling optimizer.step, (4) not calling scheduler.step at the right frequency, (5) not switching train/eval mode.",
    code: `def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, total_correct, total = 0., 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                   # 1. Clear gradients

        with torch.cuda.amp.autocast():          # 2. Forward (float16)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        scaler.scale(loss).backward()            # 3. Backward (scaled)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 4. Clip
        scaler.step(optimizer)                   # 5. Update
        scaler.update()
        # scheduler.step()  # For OneCycleLR (per-batch)

        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total         += len(y_batch)

    return total_loss / total, total_correct / total`,
    tableLinks: "model.eval() vs model.train() · Mixed Precision Training · Gradient Clipping",
  },
  {
    name: "Early Stopping & Best Model Tracking",
    lib: "torch (manual)",
    api: "torch.save(model.state_dict(), path) / model.load_state_dict(torch.load(path))",
    group: "Training Patterns",
    type: "🟢 Pattern",
    description: "Monitor a validation metric across epochs, save the model when it improves (checkpoint), and stop training if it hasn't improved for 'patience' epochs. Prevents overfitting without knowing the exact number of epochs in advance.",
    useWhen: "Every training run. Always save the best model checkpoint, not the final epoch model — final epoch is almost always overfit compared to the best validation checkpoint.",
    dontUse: "Stopping too early (patience too small). Monitoring training loss instead of validation loss (will stop as soon as training loss plateaus, even if validation is still improving).",
    pitfall: "Saving model.state_dict() is lightweight and portable. Never save the full model object with torch.save(model) — it ties the save to your exact Python/class structure and breaks on refactoring.",
    code: `class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, checkpoint_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False   # Don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded

# Load best model after training:
model.load_state_dict(torch.load('best_model.pt'))
model.eval()`,
    tableLinks: "Early Stopping Misconfiguration (Problems) · Overfitting — High Variance",
  },
  {
    name: "Gradient Accumulation",
    lib: "torch",
    api: "loss = loss / accumulation_steps; if (step+1) % accumulation_steps == 0: optimizer.step()",
    group: "Training Patterns",
    type: "🟢 Pattern",
    description: "Simulates a larger effective batch size by accumulating gradients over multiple forward passes before doing one optimizer step. Allows training transformer-scale models on GPUs with limited VRAM.",
    useWhen: "GPU VRAM is too small for your desired effective batch size. Transformer training where large batch sizes are beneficial. When batch_size=1024 is needed but only batch_size=64 fits in VRAM.",
    dontUse: "When you can already fit the desired batch size in VRAM (adds unnecessary iteration complexity).",
    pitfall: "Loss must be divided by accumulation_steps before backward to get the correct gradient magnitude. Skipping this division makes the gradient K× too large where K is the accumulation steps.",
    code: `accumulation_steps = 8  # Effective batch size = batch_size * accumulation_steps

optimizer.zero_grad()

for step, (X_batch, y_batch) in enumerate(train_loader):
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    logits = model(X_batch)
    loss   = criterion(logits, y_batch)
    loss   = loss / accumulation_steps    # ← Scale loss by accumulation factor

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()`,
    tableLinks: "Memory Errors / OOM (Problems) · Batch Size Effects (Problems)",
  },

  // ═══════════════════════════════════════════
  // SKORCH CORE
  // ═══════════════════════════════════════════
  {
    name: "NeuralNetClassifier",
    lib: "skorch",
    api: "from skorch import NeuralNetClassifier",
    group: "Skorch Core",
    type: "🟡 Skorch",
    description: "Wraps a PyTorch nn.Module to provide a full sklearn-compatible classifier interface: fit(), predict(), predict_proba(), score(). Handles the training loop, device management, batching, and epoch tracking automatically. The core entry point for multi-class classification.",
    useWhen: "Multi-class classification with PyTorch models and you want sklearn API compatibility (Pipeline, GridSearchCV, cross_val_score). Reduces boilerplate dramatically.",
    dontUse: "Binary classification with a single output node (use NeuralNetBinaryClassifier). Regression (use NeuralNetRegressor). Tasks needing very custom training loops.",
    pitfall: "Input X must be float32, y must be int64 (long). Skorch does NOT automatically cast. Always call X = X.astype(np.float32) and y = y.astype(np.int64) before fitting.",
    code: `import torch
from torch import nn
from skorch import NeuralNetClassifier

class MLP(nn.Module):
    def __init__(self, in_features=20, n_units=64, out_classes=2):
        super().__init__()
        # n_units and in_features are tunable via module__n_units, module__in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, n_units), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(n_units, out_classes)
        )
    def forward(self, x):
        return self.net(x)

net = NeuralNetClassifier(
    MLP,
    module__in_features=20,          # module__ prefix → passed to MLP.__init__
    module__n_units=64,
    max_epochs=50,
    lr=1e-3,
    batch_size=64,
    iterator_train__shuffle=True,    # iterator__ → passed to DataLoader
    train_split=None,                # Disable internal validation split (we use CV)
    verbose=0,
)

# Now it's a full sklearn estimator:
net.fit(X_train.astype(np.float32), y_train.astype(np.int64))
y_proba = net.predict_proba(X_test.astype(np.float32))`,
    tableLinks: "MLP Classifier (sklearn table) · Wrong Cross-Validation Strategy (Problems)",
  },
  {
    name: "NeuralNetBinaryClassifier",
    lib: "skorch",
    api: "from skorch import NeuralNetBinaryClassifier",
    group: "Skorch Core",
    type: "🟡 Skorch",
    description: "Specialised wrapper for binary classification with a single output neuron and sigmoid activation. Automatically uses BCEWithLogitsLoss and returns probabilities directly from a single logit output.",
    useWhen: "Binary classification where your model outputs a single logit (scalar) per sample. Slightly more convenient than NeuralNetClassifier for binary tasks.",
    dontUse: "Multi-class classification. When your model outputs two logits and you use softmax (use NeuralNetClassifier in that case).",
    pitfall: "The model's final layer must output shape (batch, 1) or (batch,) — a single scalar. Outputting two logits with NeuralNetBinaryClassifier will cause a dimension error.",
    code: `from skorch import NeuralNetBinaryClassifier

class BinaryMLP(nn.Module):
    def __init__(self, in_f=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)    # ← Single output logit (no sigmoid)
        )
    def forward(self, x):
        return self.net(x)

net = NeuralNetBinaryClassifier(
    BinaryMLP,
    module__in_f=20,
    max_epochs=50,
    lr=1e-3,
    criterion=nn.BCEWithLogitsLoss,   # Default — with pos_weight for imbalance
)

net.fit(X_train.astype(np.float32), y_train.astype(np.float32))
# y must be float32 for BCEWithLogitsLoss, not int64
proba = net.predict_proba(X_test.astype(np.float32))[:, 1]`,
    tableLinks: "BCEWithLogitsLoss · Class Imbalance (Problems) · Threshold Tuning",
  },
  {
    name: "NeuralNetRegressor",
    lib: "skorch",
    api: "from skorch import NeuralNetRegressor",
    group: "Skorch Core",
    type: "🟡 Skorch",
    description: "Wraps a PyTorch model for regression with an sklearn-compatible interface. Uses MSELoss by default. Provides fit(), predict(), and score() (returns R²). Integrates directly with sklearn Pipeline and cross_val_score.",
    useWhen: "Regression tasks with PyTorch models where you want sklearn CV, GridSearch, and Pipeline integration.",
    dontUse: "When the target is class labels (use NeuralNetClassifier). When you need custom training loop logic (write the loop manually or subclass NeuralNet).",
    pitfall: "Target y must be float32, shape (n_samples, 1) — NOT (n_samples,). The extra dimension is required for MSELoss. Use y.reshape(-1, 1).astype(np.float32).",
    code: `from skorch import NeuralNetRegressor

class RegressorMLP(nn.Module):
    def __init__(self, in_f=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)   # Single output for regression
        )
    def forward(self, x):
        return self.net(x)

net = NeuralNetRegressor(
    RegressorMLP,
    module__in_f=X_train.shape[1],
    max_epochs=100,
    lr=1e-3,
    criterion=nn.HuberLoss,    # Robust alternative to MSE
    batch_size=64,
    train_split=None,
)

# y must be (n, 1) float32:
net.fit(X_train.astype(np.float32), y_train.reshape(-1,1).astype(np.float32))
y_pred = net.predict(X_test.astype(np.float32))  # Returns (n, 1)`,
    tableLinks: "MLP Regressor (sklearn table) · Log Transform (FE table) · Wrong Evaluation Metric",
  },
  {
    name: "module__ / criterion__ / optimizer__ Parameter Prefix",
    lib: "skorch",
    api: "net = NeuralNetClassifier(MLP, module__n_units=64, optimizer__lr=0.01)",
    group: "Skorch Core",
    type: "🟡 Skorch",
    description: "Skorch uses double-underscore prefixes to pass parameters to nested components. module__ → forwarded to the nn.Module constructor. criterion__ → forwarded to the loss class. optimizer__ → overrides optimizer kwargs. iterator_train__ → DataLoader kwargs.",
    useWhen: "Setting hyperparameters of the underlying components. Tuning module architecture parameters in GridSearchCV. Passing weight tensors to loss functions or setting DataLoader options.",
    dontUse: "Don't try to pass things that should be attributes of the NeuralNet wrapper itself (like max_epochs, lr) with a prefix.",
    pitfall: "In GridSearchCV param_grid with a Pipeline: params must be prefixed with the pipeline step name: 'neuralnet__module__n_units'. Triple-underscore notation through a pipeline is a common source of confusion.",
    code: `# Direct use:
net = NeuralNetClassifier(
    MLP,
    module__n_units=64,             # → MLP(n_units=64)
    module__dropout=0.3,            # → MLP(dropout=0.3)
    criterion=nn.CrossEntropyLoss,
    criterion__label_smoothing=0.1, # → CrossEntropyLoss(label_smoothing=0.1)
    optimizer=torch.optim.AdamW,
    optimizer__weight_decay=0.01,   # → AdamW(weight_decay=0.01)
    iterator_train__shuffle=True,   # → DataLoader(shuffle=True)
    iterator_train__num_workers=4,  # → DataLoader(num_workers=4)
)

# GridSearchCV param_grid — prefix module architecture params:
param_grid = {
    'module__n_units':   [32, 64, 128],
    'module__dropout':   [0.1, 0.3, 0.5],
    'lr':                [1e-3, 1e-4],
    'optimizer__weight_decay': [1e-4, 1e-2],
}`,
    tableLinks: "Hyperparameter Tuning Without Principled Strategy (Problems)",
  },

  // ═══════════════════════════════════════════
  // SKORCH + SKLEARN
  // ═══════════════════════════════════════════
  {
    name: "GridSearchCV / RandomizedSearchCV with Skorch",
    lib: "skorch + sklearn",
    api: "GridSearchCV(net, param_grid, cv=5, scoring='roc_auc').fit(X, y)",
    group: "Skorch + sklearn",
    type: "🟡 Skorch",
    description: "Because skorch estimators implement the sklearn estimator interface, they work directly with GridSearchCV, RandomizedSearchCV, and Optuna's sklearn integration — allowing hyperparameter search over both model architecture and training parameters in one unified call.",
    useWhen: "Systematic hyperparameter tuning of PyTorch models without writing custom tuning loops. Works best with Optuna's cross_val_score-based objective for Bayesian search.",
    dontUse: "GridSearchCV for large search spaces (exponential combinations). Use RandomizedSearchCV or Optuna instead. Avoid training_split inside net when using CV — set train_split=None.",
    pitfall: "Each CV fold with max_epochs=50 trains a full neural net from scratch. With 5 folds × 20 parameter combinations = 100 full training runs. Skorch's warm_start=False (default) means each starts fresh, which is slow for large models.",
    code: `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'module__n_units':   [32, 64, 128],
    'lr':                [1e-3, 5e-4, 1e-4],
    'max_epochs':        [30, 50],
    'module__dropout':   [0.2, 0.4],
}

# Use train_split=None so outer CV handles validation:
net = NeuralNetClassifier(MLP, train_split=None, verbose=0)

gs = GridSearchCV(net, param_grid, cv=5, scoring='roc_auc', n_jobs=1)
# n_jobs=1 because PyTorch manages its own parallelism

gs.fit(X_train.astype(np.float32), y_train.astype(np.int64))
print(f"Best ROC-AUC: {gs.best_score_:.4f}")
print(f"Best params:  {gs.best_params_}")`,
    tableLinks: "Hyperparameter Tuning Without Principled Strategy (Problems) · Overfitting the Validation Set (Problems)",
  },
  {
    name: "sklearn Pipeline with Skorch",
    lib: "skorch + sklearn",
    api: "Pipeline([('scaler', StandardScaler()), ('net', NeuralNetClassifier(...))])",
    group: "Skorch + sklearn",
    type: "🟡 Skorch",
    description: "Combining sklearn preprocessing steps with a skorch neural net in a Pipeline ensures correct fit/transform separation across CV folds — preventing preprocessing leakage. The pipeline can be passed to cross_val_score, GridSearchCV, and SHAP.",
    useWhen: "Always when combining preprocessing (scaling, imputation, encoding) with a neural net. The Pipeline guarantees the scaler is fit on training folds only — the most important leakage prevention tool.",
    dontUse: "When using PyTorch's own transforms or albumentations (image pipelines) — those go inside the Dataset/DataLoader, not the sklearn Pipeline.",
    pitfall: "In a Pipeline, hyperparameters must be prefixed with the step name: 'net__module__n_units' (step_name + double underscore + skorch param). Forgetting this prefix causes a ValueError that's easy to misdiagnose.",
    code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
    ('net',    NeuralNetClassifier(
        MLP,
        module__in_features=20,
        max_epochs=50,
        lr=1e-3,
        train_split=None,
        verbose=0,
    )),
])

# Preprocessing is automatically fit on train fold, transform applied to both:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"OOF AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# GridSearchCV with pipeline — note triple prefix: step__skorch_param__module_param:
param_grid = {
    'net__module__n_units':   [64, 128],
    'net__lr':                [1e-3, 1e-4],
}`,
    tableLinks: "Preprocessing Leakage (Fit on Full Data) (Problems) · sklearn Pipeline (sklearn table)",
  },
  {
    name: "Optuna + Skorch (Bayesian Hyperparameter Search)",
    lib: "optuna + skorch + sklearn",
    api: "study.optimize(objective, n_trials=100)",
    group: "Skorch + sklearn",
    type: "🟡 Skorch",
    description: "Optuna's Bayesian optimisation paired with skorch and cross_val_score for principled, efficient hyperparameter search. Bayesian search finds good configurations in 10× fewer trials than grid search by learning which regions of hyperparameter space are promising.",
    useWhen: "Any serious hyperparameter tuning of PyTorch/skorch models. Always prefer Optuna over GridSearchCV for neural nets due to the large, continuous search spaces.",
    dontUse: "Very quick experiments where manual tuning of LR and hidden size is faster. When training is so slow that 50 Optuna trials would take days — use a random sample instead.",
    pitfall: "Always set n_jobs=1 in cross_val_score when using PyTorch — PyTorch manages its own threading and parallel CV folds conflict with it.",
    code: `import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    net = NeuralNetClassifier(
        MLP,
        module__n_units   = trial.suggest_int('n_units', 32, 256),
        module__dropout   = trial.suggest_float('dropout', 0.1, 0.6),
        lr                = trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        optimizer         = torch.optim.AdamW,
        optimizer__weight_decay = trial.suggest_float('wd', 1e-5, 1e-1, log=True),
        max_epochs        = 50,
        train_split       = None,
        verbose           = 0,
    )
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('net',   net),
    ])
    scores = cross_val_score(pipeline, X_train.astype(np.float32),
                              y_train.astype(np.int64),
                              cv=5, scoring='roc_auc', n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
print(f"Best AUC: {study.best_value:.5f}")`,
    tableLinks: "Hyperparameter Tuning Without Principled Strategy (Problems) · Nested CV Neglect",
  },

  // ═══════════════════════════════════════════
  // SKORCH CALLBACKS
  // ═══════════════════════════════════════════
  {
    name: "EarlyStopping Callback",
    lib: "skorch.callbacks",
    api: "from skorch.callbacks import EarlyStopping",
    group: "Skorch Callbacks",
    type: "🟡 Skorch",
    description: "Monitors a validation metric and stops training when it hasn't improved for 'patience' epochs. Works identically to manual early stopping but is declared once as a callback. Requires train_split to be set (non-None) so Skorch creates an internal validation set.",
    useWhen: "Always when using Skorch for production-quality training. Replace max_epochs=50 with max_epochs=1000 + EarlyStopping(patience=20) for a more principled training regime.",
    dontUse: "When using Skorch inside GridSearchCV/cross_val_score with train_split=None — there's no internal validation set to monitor, so EarlyStopping has nothing to track.",
    pitfall: "EarlyStopping requires an internal validation split (train_split is not None). When using cross_val_score, this means 5-fold CV × internal validation = nested CV. Set train_split=None and use cross_val_score's CV exclusively to avoid confusion.",
    code: `from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint
from skorch.dataset import ValidSplit

net = NeuralNetClassifier(
    MLP,
    module__n_units=128,
    max_epochs=500,                      # High ceiling — early stopping will trigger
    lr=1e-3,
    train_split=ValidSplit(0.2, random_state=42),  # 20% internal validation
    callbacks=[
        EarlyStopping(
            monitor='valid_loss',         # Metric to watch
            patience=20,                  # Stop after 20 epochs without improvement
            lower_is_better=True,         # valid_loss should decrease
        ),
        Checkpoint(
            monitor='valid_acc_best',     # Save best model by validation accuracy
            f_params='best_model.pt',
        ),
    ],
    verbose=1,
)`,
    tableLinks: "Early Stopping Misconfiguration (Problems) · Overfitting — High Variance",
  },
  {
    name: "LRScheduler Callback",
    lib: "skorch.callbacks",
    api: "from skorch.callbacks import LRScheduler",
    group: "Skorch Callbacks",
    type: "🟡 Skorch",
    description: "Integrates any torch.optim.lr_scheduler with Skorch's training loop. Handles calling scheduler.step() at the right time (per epoch or per batch). Supports all PyTorch schedulers through the policy parameter.",
    useWhen: "Adding learning rate scheduling to any Skorch model. Should almost always be used with any non-trivial training run.",
    dontUse: "When using ReduceLROnPlateau — monitor must be explicitly set. Don't set step_every='batch' for epoch-based schedulers (CosineAnnealing, StepLR).",
    pitfall: "LRScheduler with ReduceLROnPlateau requires monitor='valid_loss' — it won't know what metric to plateau-detect without this. Other schedulers don't need monitor.",
    code: `from skorch.callbacks import LRScheduler

# CosineAnnealingLR (per epoch):
net = NeuralNetClassifier(
    MLP,
    callbacks=[
        LRScheduler(
            policy=torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=100,              # kwargs for the scheduler constructor
            eta_min=1e-7,
        ),
    ],
)

# ReduceLROnPlateau (monitors val loss):
net = NeuralNetClassifier(
    MLP,
    callbacks=[
        LRScheduler(
            policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
            monitor='valid_loss',   # ← Required for ReduceLROnPlateau
            factor=0.5,
            patience=5,
        ),
    ],
)`,
    tableLinks: "LR Schedulers · Learning Rate Too High / Too Low (Problems)",
  },
  {
    name: "GradientNormClipping Callback",
    lib: "skorch.callbacks",
    api: "from skorch.callbacks import GradientNormClipping",
    group: "Skorch Callbacks",
    type: "🟡 Skorch",
    description: "Applies gradient norm clipping inside Skorch's training loop without manually modifying the training code. Wraps torch.nn.utils.clip_grad_norm_.",
    useWhen: "Any Skorch-based RNN, LSTM, or Transformer training. When loss becomes NaN during Skorch training.",
    dontUse: "When you're already managing gradient clipping manually in a custom training loop outside Skorch.",
    pitfall: "GradientNormClipping should be listed before other callbacks that use gradients in the callbacks list — callback order matters.",
    code: `from skorch.callbacks import GradientNormClipping, ProgressBar

net = NeuralNetClassifier(
    LSTMClassifier,
    max_epochs=100,
    lr=1e-3,
    callbacks=[
        GradientNormClipping(gradient_clip_value=1.0),  # Clip before optimizer.step()
        ProgressBar(),                                   # Optional progress bar
    ],
)`,
    tableLinks: "Exploding Gradients (DL Problems) · Gradient Clipping",
  },

  // ═══════════════════════════════════════════
  // ARCHITECTURE PATTERNS
  // ═══════════════════════════════════════════
  {
    name: "MLP for Tabular Data",
    lib: "torch.nn",
    api: "nn.Sequential(Linear+BN+ReLU+Dropout) × n layers",
    group: "Architecture Patterns",
    type: "🟢 Pattern",
    description: "Fully-connected neural network for tabular features. Typically 2-4 hidden layers with BatchNorm + ReLU + Dropout blocks. Competitive with GBMs when features are scaled and the dataset has > 10k rows. Benefit: captures non-linear interactions automatically.",
    useWhen: "Tabular classification/regression when GBMs plateau. Entity embedding for high-cardinality categoricals. Competition second-layer model for ensembling diversity.",
    dontUse: "Very small datasets (< 5k rows) — GBMs generalize better. When interpretability is critical. As first model to try (always baseline with GBMs first).",
    pitfall: "Without BatchNorm, MLPs on tabular data are very sensitive to feature scaling. Always include BatchNorm or ensure input is StandardScaled.",
    code: `class TabularMLP(nn.Module):
    def __init__(self, in_features, layer_sizes=(256,128,64), n_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for size in layer_sizes:
            layers += [
                nn.Linear(prev, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = size
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Usage with Skorch:
net = NeuralNetClassifier(
    TabularMLP,
    module__in_features=X_train.shape[1],
    module__layer_sizes=(256, 128, 64),
    module__n_classes=len(np.unique(y)),
    lr=1e-3, max_epochs=100, batch_size=256, train_split=None,
)`,
    tableLinks: "MLP Classifier (sklearn table) · Skorch Core · Feature Scale Sensitivity",
  },
  {
    name: "1D CNN for Sequences / Time Series",
    lib: "torch.nn",
    api: "nn.Conv1d(in_channels, out_channels, kernel_size) + nn.AdaptiveMaxPool1d",
    group: "Architecture Patterns",
    type: "🟢 Pattern",
    description: "1D Convolutional network that processes sequences as 1D signals. Captures local temporal patterns efficiently and parallelises much better than LSTM. Often outperforms LSTM on time series classification with shorter/fixed-length sequences.",
    useWhen: "Time series classification/regression with fixed-length windows. Faster alternative to LSTM for sequence modelling. Text classification (Kim-style CNN).",
    dontUse: "Very long sequences with important long-range dependencies (Transformer is better). Variable-length sequences without padding (use pack_padded_sequence for LSTM instead).",
    pitfall: "Input shape must be (batch, channels, length) — note channels before length. If your sequence is (batch, length, features), transpose it: x = x.transpose(1, 2) before Conv1d.",
    code: `class TemporalCNN(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters=64):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, n_filters,   kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(n_filters),
            nn.Conv1d(n_filters,   n_filters*2, kernel_size=5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(n_filters*2),
            nn.Conv1d(n_filters*2, n_filters*4, kernel_size=7, padding=3),
            nn.ReLU(), nn.BatchNorm1d(n_filters*4),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)   # Global max pool → fixed size
        self.fc   = nn.Linear(n_filters*4, n_classes)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = self.pool(self.convs(x)).squeeze(-1)  # → (batch, n_filters*4)
        return self.fc(x)`,
    tableLinks: "Time Series Features (FE table) · Lag Features · ACF/PACF",
  },

  // ═══════════════════════════════════════════
  // TRANSFER LEARNING
  // ═══════════════════════════════════════════
  {
    name: "Loading Pre-trained Models (torchvision / HuggingFace)",
    lib: "torchvision · transformers",
    api: "torchvision.models.resnet50(weights='DEFAULT') / AutoModel.from_pretrained('bert-base')",
    group: "Transfer Learning",
    type: "🟢 Pattern",
    description: "Loading neural network weights pre-trained on large datasets (ImageNet for vision, Wikipedia for NLP). Provides a powerful feature extractor from day one — dramatically reduces training data requirements and training time.",
    useWhen: "Any image task with < 100k samples — always use transfer learning. NLP tasks — always start from a pre-trained transformer (BERT, RoBERTa). Any task where the source and target domains share low-level features.",
    dontUse: "Highly domain-specific data where the pre-trained features are unrelated (e.g., medical X-ray with ImageNet features may still help but less so). Embedded/mobile models where the architecture must be from scratch.",
    pitfall: "The final classification head of pre-trained models has the number of ImageNet (1000) or source-task classes. Always replace this head with a new one matching your number of classes.",
    code: `import torchvision.models as models

# Load ResNet50 with ImageNet weights, replace the head:
backbone = models.resnet50(weights='DEFAULT')  # weights='DEFAULT' = best ImageNet weights
n_features  = backbone.fc.in_features          # 2048 for ResNet50
backbone.fc = nn.Linear(n_features, n_classes) # Replace with your head

# For NLP — HuggingFace transformer:
from transformers import AutoModel
bert = AutoModel.from_pretrained('bert-base-uncased')
class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert    = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(768, n_classes)  # BERT hidden size = 768
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]      # CLS token
        return self.fc(self.dropout(cls))`,
    tableLinks: "CNN Embeddings / Transfer Learning (FE table) · Contextual Sentence Embeddings (FE table)",
  },
  {
    name: "Freezing Layers & Fine-Tuning Strategy",
    lib: "torch",
    api: "param.requires_grad = False / Discriminative LR",
    group: "Transfer Learning",
    type: "🟢 Pattern",
    description: "Selective layer freezing controls which pre-trained weights are updated during training. Phase 1: freeze the backbone, train only the new head (fast, few epochs). Phase 2: unfreeze the full network with a very small LR (fine-tuning). Prevents catastrophic forgetting.",
    useWhen: "Any transfer learning setup. Always fine-tune in two phases: head-only first, then full network with discriminative LR.",
    dontUse: "Freezing the entire model forever — the model can't adapt to domain-specific features. Fine-tuning with the same LR everywhere — lower layers should have a lower LR than the head.",
    pitfall: "Forgetting to re-enable gradients after Phase 1. Always check which parameters are being optimised by printing [n for n, p in model.named_parameters() if p.requires_grad].",
    code: `# Phase 1: Freeze backbone, train only head (5-10 epochs)
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc.requires_grad_(True)  # Only train the head

optimizer = torch.optim.Adam(backbone.fc.parameters(), lr=1e-3)

# Phase 2: Unfreeze everything with discriminative LR
for param in backbone.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': backbone.layer1.parameters(), 'lr': 1e-5},   # Earliest layers
    {'params': backbone.layer2.parameters(), 'lr': 2e-5},
    {'params': backbone.layer3.parameters(), 'lr': 5e-5},
    {'params': backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': backbone.fc.parameters(),     'lr': 1e-3},   # Head
], weight_decay=1e-4)`,
    tableLinks: "Overfitting — High Variance (Problems) · AdamW · Weight Decay",
  },

  // ═══════════════════════════════════════════
  // DL-SPECIFIC PROBLEMS
  // ═══════════════════════════════════════════
  {
    name: "Vanishing Gradients",
    lib: "torch",
    api: "Diagnosed via gradient norm monitoring per layer",
    group: "DL-Specific Problems",
    type: "🔴 Problem",
    description: "Gradients become exponentially small during backpropagation through deep networks, causing early layers to learn extremely slowly or not at all. Symptom: early layer gradient norms are orders of magnitude smaller than later layers.",
    useWhen: "Any deep network (> 5 layers). Networks with sigmoid/tanh activations. RNNs without LSTM/GRU gating.",
    dontUse: "n/a — this is a problem to detect and fix, not a technique to apply.",
    pitfall: "Vanishing gradients are silent — the model trains without errors but early layers learn nothing. The model appears to converge but to a poor solution.",
    code: `# Diagnose: check gradient norms per layer after backward()
model.zero_grad()
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.2e}")
# If early layers show 1e-10 while last layers show 1e-3 → vanishing

# FIXES:
# 1. Switch to ReLU/GELU (non-saturating activations)
# 2. Add BatchNorm/LayerNorm between layers
# 3. Use residual connections (ResNet-style)
# 4. Use He/Kaiming weight initialisation:
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
# 5. Use LSTM/GRU instead of vanilla RNN
# 6. Reduce network depth`,
    tableLinks: "Vanishing Gradients (Problems) · BatchNorm / LayerNorm · Activation Functions",
  },
  {
    name: "Exploding Gradients / NaN Loss",
    lib: "torch",
    api: "torch.nn.utils.clip_grad_norm_ / torch.cuda.amp.GradScaler",
    group: "DL-Specific Problems",
    type: "🔴 Problem",
    description: "Gradients grow exponentially during backpropagation, causing weight updates that are so large they make the loss diverge to NaN. Common in RNNs, deep networks, and during mixed-precision training when float16 overflows.",
    useWhen: "Any RNN/LSTM training, transformer training, or when loss suddenly becomes NaN.",
    dontUse: "n/a",
    pitfall: "NaN loss caused by a data outlier in a batch is harder to diagnose than architectural exploding gradients. Always check your input data for NaN/inf values before training.",
    code: `# Detect: check for NaN loss
if torch.isnan(loss):
    print("NaN loss detected!")
    # Inspect batch for NaN inputs:
    print(f"X NaN: {torch.isnan(X_batch).any()}")
    print(f"y NaN: {torch.isnan(y_batch.float()).any()}")

# Fix 1: Gradient clipping (always for RNN/Transformer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Fix 2: Reduce learning rate by 10×

# Fix 3: Check data for NaN/inf BEFORE training:
assert not np.isnan(X_train).any(), "NaN in input features"
assert not np.isinf(X_train).any(), "Inf in input features"

# Fix 4: For AMP NaN — use dynamic scaling and check:
scaler = GradScaler()           # Dynamic scaling (default)
if scaler.get_scale() < 1:     # Scale underflowed → something is wrong
    print("WARNING: GradScaler scale collapsed")`,
    tableLinks: "Exploding Gradients (Problems) · Mixed Precision Training · Learning Rate",
  },
  {
    name: "Dead Neurons (Dying ReLU)",
    lib: "torch",
    api: "nn.LeakyReLU(0.01) / nn.ELU() / correct weight initialisation",
    group: "DL-Specific Problems",
    type: "🔴 Problem",
    description: "A ReLU neuron 'dies' when its pre-activation output is always negative, making the gradient permanently zero — that neuron never updates again. Can kill a significant fraction of neurons, especially with high learning rates or poor initialisation.",
    useWhen: "Any network using ReLU activations, especially with SGD and high learning rates.",
    dontUse: "n/a",
    pitfall: "Dead neurons are completely invisible during training — loss decreases normally but the effective model capacity is reduced. The only way to detect them is to explicitly count neurons with zero activation.",
    code: `# Detect: count dead neurons after training
def count_dead_neurons(model, X_sample):
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn

    hooks = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
             if isinstance(m, nn.ReLU)]
    with torch.no_grad():
        model(X_sample.to(device))
    for h in hooks: h.remove()

    for name, act in activations.items():
        dead_frac = (act == 0).float().mean().item()
        if dead_frac > 0.3:
            print(f"⚠️  {name}: {dead_frac:.1%} dead neurons")

# FIXES:
# 1. Use LeakyReLU(0.01) — small negative slope keeps neurons alive
# 2. Use ELU — smooth negative region
# 3. Use GELU (most modern architectures)
# 4. Use Kaiming He initialisation (default in nn.Linear but verify)
# 5. Reduce learning rate`,
    tableLinks: "Activation Functions · Vanishing Gradients · Overfitting DL-Specific",
  },
  {
    name: "Overfitting (DL-Specific)",
    lib: "torch",
    api: "Dropout + weight_decay + early_stopping + data augmentation",
    group: "DL-Specific Problems",
    type: "🔴 Problem",
    description: "Neural networks have millions of parameters and can perfectly memorise training data. Unlike sklearn models, DL overfitting requires a specific combination of regularisation techniques: dropout, weight decay, early stopping, data augmentation, and sometimes label smoothing.",
    useWhen: "Any neural network training where val_loss > train_loss by a significant margin.",
    dontUse: "n/a",
    pitfall: "Applying all regularisation techniques simultaneously from the start makes it impossible to diagnose which one is effective. Add them one at a time and check CV score after each.",
    code: `# DL regularisation checklist — apply in this order:
# 1. Weight decay (always on — no downside)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 2. Dropout (add to model architecture)
self.dropout = nn.Dropout(p=0.3)  # After activations in hidden layers

# 3. Early stopping (via skorch callback or manual)
early_stop = EarlyStopping(patience=20, monitor='valid_loss')

# 4. Label smoothing (for classification)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 5. Data augmentation (for images/text)
augmentation = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightness()])

# 6. Reduce model capacity (fewer layers/units)
# 7. Get more data (best solution if available)

# Diagnostic: plot train vs val loss per epoch
# Large, growing gap → overfitting
# Both curves high → underfitting`,
    tableLinks: "Overfitting — High Variance (Problems) · Dropout · Mixup Augmentation",
  },
  {
    name: "Learning Rate Finder",
    lib: "torch-lr-finder · fastai",
    api: "from torch_lr_finder import LRFinder",
    group: "DL-Specific Problems",
    type: "🟠 Diagnostic",
    description: "Runs a short training pass where the learning rate is increased exponentially from a very small value to a large one, tracking the loss. The optimal LR is typically just before the loss starts to increase sharply — the steepest downward slope region.",
    useWhen: "Whenever you start training a new architecture. Removes all guesswork from initial LR selection. Run it before every training run on new data or architectures.",
    dontUse: "Very large models where even a short LR finder run is prohibitively expensive. When using OneCycleLR (you still need a max_lr estimate, but use it directly as max_lr).",
    pitfall: "The optimal LR from the finder is for a single epoch. For longer training, multiply by 0.1-0.3 to find a stable starting LR.",
    code: `# pip install torch-lr-finder
from torch_lr_finder import LRFinder

model     = MLP(in_features=20, n_units=128, out_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
criterion = nn.CrossEntropyLoss()

finder = LRFinder(model, optimizer, criterion, device=device)
finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode='exp')
finder.plot()       # Look for the steepest downward slope
finder.reset()      # Reset model and optimizer to initial state

# The plot shows loss vs log(LR)
# Good starting LR = 10× smaller than the LR at minimum loss`,
    tableLinks: "Learning Rate Too High / Too Low (Problems) · OneCycleLR · ReduceLROnPlateau",
  },

  // ═══════════════════════════════════════════
  // ALTERNATIVE DL LIBRARIES
  // ═══════════════════════════════════════════
  {
    name: "pytorch-tabnet (TabNet)",
    lib: "pytorch-tabnet",
    api: "from pytorch_tabnet.tab_model import TabNetClassifier",
    group: "Alternative DL Libraries",
    type: "🟣 Component",
    description: "Attention-based neural network specifically designed for tabular data. Performs sequential feature selection at each step using a learnable attention mechanism. Provides built-in feature importances and interpretability. Can compete with GBMs on tabular data.",
    useWhen: "Tabular classification/regression when you want a neural alternative to GBMs with built-in interpretability. When feature selection within the model is desirable. Competition ensembling — TabNet provides diversity from GBMs.",
    dontUse: "Small datasets (< 5k rows) — TabNet needs more data than GBMs to train reliably. When training speed is critical (TabNet is slower than LightGBM).",
    pitfall: "TabNet is sensitive to the n_steps hyperparameter (number of sequential attention steps). Default (3) may be insufficient for high-dimensional data. Tune n_steps in [3, 5, 8].",
    code: `from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np

clf = TabNetClassifier(
    n_d=64,              # Width of prediction step layers
    n_a=64,              # Width of attention step layers
    n_steps=5,           # Number of sequential attention steps
    gamma=1.3,           # Coefficient for feature reusage in attention
    n_independent=2,     # Number of independent GLU layers in each step
    n_shared=2,          # Number of shared GLU layers
    lambda_sparse=1e-3,  # Sparsity regularisation on feature selection
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',  # 'sparsemax' or 'entmax'
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['auc'],
    patience=20,
    max_epochs=200,
)
preds = clf.predict_proba(X_test)[:, 1]

# Feature importances (built-in):
feat_imp = clf.feature_importances_`,
    tableLinks: "Ensemble Diversity Collapse (Problems) · MLP for Tabular Data · SHAP Values",
  },
  {
    name: "PyTorch Lightning",
    lib: "lightning",
    api: "class LitModel(L.LightningModule): training_step / configure_optimizers",
    group: "Alternative DL Libraries",
    type: "🟢 Pattern",
    description: "High-level training framework that abstracts the PyTorch training loop into a structured LightningModule class. Handles device placement, distributed training, logging, gradient clipping, mixed precision, and checkpointing — all via a clean interface. Complements (not replaces) Skorch.",
    useWhen: "Complex training scenarios: multi-GPU, distributed training, complex logging with W&B/TensorBoard, reproducible research. Large models (vision, NLP) where the training infrastructure is as complex as the model.",
    dontUse: "Simple tabular tasks where Skorch + sklearn pipeline is more convenient. Quick experiments where the overhead of the LightningModule structure is overkill.",
    pitfall: "Lightning and Skorch solve different problems: Lightning is a training framework for complex DL training loops; Skorch is an sklearn compatibility layer. For sklearn integration (cross_val_score, GridSearchCV), Skorch is the right tool. Use Lightning for complex training, Skorch for sklearn integration.",
    code: `import lightning as L

class LitClassifier(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.01):
        super().__init__()
        self.model     = model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr        = lr
        self.wd        = weight_decay

    def training_step(self, batch, batch_idx):
        X, y   = batch
        logits = self.model(X)
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y   = batch
        logits = self.model(X)
        loss   = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sched]

trainer = L.Trainer(
    max_epochs=100,
    gradient_clip_val=1.0,
    precision='16-mixed',   # AMP
    callbacks=[L.pytorch.callbacks.EarlyStopping('val_loss', patience=20)],
)
trainer.fit(LitClassifier(model), train_loader, val_loader)`,
    tableLinks: "Mixed Precision Training · Gradient Clipping · EarlyStopping · Transfer Learning",
  },
];

// ─── COMPONENT ───────────────────────────────────────────────
const ALL_GROUPS    = ["All", ...Object.keys(GROUP_ACCENT)];
const ALL_TYPES     = ["All", ...Object.keys(TYPE_STYLE)];

function TypeBadge({ t }) {
  const s = TYPE_STYLE[t] || { bg:"var(--border-subtle)", color:"#888", border:"#888" };
  return <span style={{ display:"inline-block", padding:"2px 8px", borderRadius:"4px", fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.04em", background:s.bg, color:s.color, border:`1px solid ${s.border}44`, whiteSpace:"nowrap" }}>{t}</span>;
}
function LibBadge({ text }) {
  return <span style={{ display:"inline-block", padding:"1px 6px", borderRadius:"3px", fontSize:"0.61rem", fontWeight:600, background:"#0d0d1c", color:"#5a5a88", border:"1px solid var(--border-default)", whiteSpace:"nowrap", margin:"1px" }}>{text}</span>;
}
function Card({ icon, label, text, accent }) {
  return (
    <div style={{ padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:`1px solid ${accent}1e` }}>

      <div style={{ fontSize:"0.62rem", fontWeight:700, letterSpacing:"0.09em", color:accent, textTransform:"uppercase", marginBottom:"5px" }}>{icon} {label}</div>
      <div style={{ fontSize:"0.78rem", color:"var(--text-primary)", lineHeight:1.6 }}>{text}</div>
    </div>
  );
}
function CodeBlock({ code }) {
  const [show, setShow] = useState(false);
  if (!code) return null;
  return (
    <div style={{ marginTop:"10px" }}>
      <button onClick={() => setShow(s => !s)} style={{ background:"var(--bg-surface)", border:"1px solid var(--border-default)", borderRadius:"5px", padding:"4px 12px", fontSize:"0.68rem", color:"#5a5a88", cursor:"pointer", fontFamily:"var(--font-mono)" }}>
        {show ? "▲ hide code" : "▶ show code"}
      </button>
      {show && (
        <pre style={{ margin:"6px 0 0", padding:"14px 16px", background:"var(--bg-surface)", border:"1px solid var(--border-subtle)", borderRadius:"6px", fontSize:"0.7rem", fontFamily:"var(--font-mono)", color:"#8aaccc", overflowX:"auto", lineHeight:1.7, whiteSpace:"pre" }}>
          {code}
        </pre>
      )}
    </div>
  );
}

const tdS = { padding:"10px 13px", verticalAlign:"middle", borderBottom:"1px solid var(--bg-surface)", color:"var(--text-primary)", fontSize:"0.81rem" };
const thS = { padding:"10px 13px", textAlign:"left", fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:"var(--text-tertiary)", borderBottom:"2px solid var(--bg-overlay)", background:"var(--bg-base)", position:"sticky", zIndex:5 };

function Row({ item, idx }) {
  const [open, setOpen] = useState(false);
  const accent = GROUP_ACCENT[item.group] || "#888";
  return (
    <>
      <tr onClick={() => setOpen(o=>!o)}
        onMouseEnter={e => e.currentTarget.style.background="var(--bg-elevated)"}
        onMouseLeave={e => e.currentTarget.style.background=idx%2===0?"var(--bg-surface)":"var(--bg-surface)"}
        style={{ cursor:"pointer", background:idx%2===0?"var(--bg-surface)":"var(--bg-surface)", borderLeft:`3px solid ${accent}`, transition:"background 0.12s" }}>
        <td style={tdS}>{open?"▾":"▸"}</td>
        <td style={{ ...tdS, fontWeight:700, color:accent, fontSize:"0.87rem" }}>{item.name}</td>
        <td style={tdS}>{item.lib.split("·").slice(0,2).map(l=><LibBadge key={l} text={l.trim()}/>)}</td>
        <td style={tdS}><TypeBadge t={item.type}/></td>
        <td style={{ ...tdS, fontFamily:"var(--font-mono)", fontSize:"0.67rem", color:"#32326a", maxWidth:"200px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{item.api}</td>
      </tr>
      {open && (
        <tr style={{ background:"var(--bg-surface)", borderLeft:`3px solid ${accent}` }}>
          <td colSpan={5} style={{ padding:"0 0 0 16px" }}>
            <div style={{ padding:"14px 16px 14px 0" }}>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                <Card icon="📋" label="What It Does" text={item.description} accent={accent}/>
                <Card icon="✅" label="Use When" text={item.useWhen} accent="#4ade80"/>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                <Card icon="⛔" label="Don't Use When" text={item.dontUse} accent="#f87171"/>
                <Card icon="⚠️" label="Common Pitfall" text={item.pitfall} accent="#fb923c"/>
              </div>
              <CodeBlock code={item.code}/>
              {item.tableLinks && (
                <div style={{ marginTop:"10px", padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:"1px solid #818cf822" }}>
                  <div style={{ fontSize:"0.62rem", fontWeight:700, letterSpacing:"0.09em", color:"#818cf8", textTransform:"uppercase", marginBottom:"6px" }}>🔗 See Also</div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:"4px" }}>
                    {item.tableLinks.split("·").map(t => (
                      <span key={t} style={{ padding:"2px 8px", borderRadius:"12px", fontSize:"0.63rem", fontWeight:600, background:"#818cf818", color:"#818cf8", border:"1px solid #818cf833" }}>{t.trim()}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Torch_app() {
  const [search, setSearch]   = useState("");
  const [grp,    setGrp]      = useState("All");
  const [type,   setType]     = useState("All");

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return ITEMS.filter(it => {
      const mg = grp  === "All" || it.group === grp;
      const mt = type === "All" || it.type  === type;
      const mq = !q || [it.name, it.group, it.lib, it.description, it.useWhen, it.pitfall, it.tableLinks||""].some(s => s.toLowerCase().includes(q));
      return mg && mt && mq;
    });
  }, [search, grp, type]);

  const groups = useMemo(() => {
    const g = {};
    for (const it of filtered) { if (!g[it.group]) g[it.group]=[]; g[it.group].push(it); }
    return g;
  }, [filtered]);

  return (
    <div>
    <Header />

        <div style={{ fontFamily:"var(--font-body)", background:"var(--bg-base)", minHeight:"100vh", color:"var(--text-primary)" }}>
      
      {/* HEADER */}
      <div style={{ position:"sticky", top:0, zIndex:10, background:"rgba(14,13,12,0.93)", backdropFilter:"blur(10px)", borderBottom:"1px solid var(--border-faint)", padding:"10px 20px 8px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:"14px", marginBottom:"8px", flexWrap:"wrap" }}>
          <div>
            <div style={{ fontSize:"1.05rem", fontWeight:700, color:"#fff", letterSpacing:"-0.02em" }}>
              <span style={{ color:"#fb923c" }}>PyTorch</span>
              <span style={{ color:"var(--text-dim)", margin:"0 6px" }}>·</span>
              <span style={{ color:"#facc15" }}>Skorch</span>
              <span style={{ color:"var(--text-dim)", margin:"0 6px" }}>·</span>
              <span style={{ color:"#94a3b8" }}>Deep Learning Reference</span>
            </div>
            <div style={{ fontSize:"0.62rem", color:"var(--text-dim)", letterSpacing:"0.05em", marginTop:"1px" }}>
              {ITEMS.length} entries · 16 categories · click row to expand · code snippet inside every entry
            </div>
          </div>
          <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search: optimizer, loss, dropout, LSTM, skorch…"
            style={{ flex:1, minWidth:"180px", maxWidth:"320px", marginLeft:"auto", background:"#0c0c1c", border:"1px solid var(--border-default)", borderRadius:"6px", padding:"7px 11px", color:"var(--text-primary)", fontSize:"0.79rem", outline:"none" }}/>
          <div style={{ fontSize:"0.68rem", color:"var(--text-dim)", whiteSpace:"nowrap" }}>{filtered.length} shown</div>
        </div>
        <div style={{ display:"flex", gap:"4px", flexWrap:"wrap", marginBottom:"4px" }}>
          <span style={{ fontSize:"0.6rem", color:"var(--text-dim)", alignSelf:"center", marginRight:"4px" }}>TYPE:</span>
          {ALL_TYPES.map(t => {
            const active = type===t; const s = TYPE_STYLE[t]||{color:"#666",border:"#666",bg:"#111"};
            return <button key={t} onClick={()=>setType(t)} style={{ padding:"2px 9px", borderRadius:"4px", fontSize:"0.66rem", fontWeight:active?700:400, border:active?`1px solid ${s.border}`:"1px solid var(--border-subtle)", background:active?s.bg:"transparent", color:active?s.color:"var(--text-tertiary)" }}>{t}</button>;
          })}
        </div>
        <div style={{ display:"flex", gap:"4px", flexWrap:"wrap" }}>
          <span style={{ fontSize:"0.6rem", color:"var(--text-dim)", alignSelf:"center", marginRight:"4px" }}>GROUP:</span>
          {ALL_GROUPS.map(g => {
            const active = grp===g; const accent = GROUP_ACCENT[g]||"#666";
            return <button key={g} onClick={()=>setGrp(g)} style={{ padding:"2px 9px", borderRadius:"4px", fontSize:"0.65rem", fontWeight:active?700:400, border:active?`1px solid ${accent}`:"1px solid var(--border-subtle)", background:active?`${accent}22`:"transparent", color:active?accent:"var(--text-tertiary)" }}>{g}</button>;
          })}
        </div>
      </div>

      <div style={{ overflowX:"auto", marginTop:"8px" }}>
        <table style={{ width:"100%", borderCollapse:"collapse", minWidth:"820px" }}>
          <thead>
            <tr>
              <th style={{...thS,width:"24px"}}></th>
              <th style={thS}>Name</th>
              <th style={thS}>Library</th>
              <th style={thS}>Type</th>
              <th style={{...thS,maxWidth:"200px"}}>API</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([group, items]) => (
              <>
                <tr key={group+"_h"}>
                  <td colSpan={5} style={{ padding:"8px 14px 4px", fontSize:"0.65rem", fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", color:GROUP_ACCENT[group]||"#5555a0", background:"var(--bg-base)", borderTop:"2px solid var(--bg-elevated)", borderBottom:"1px solid var(--bg-elevated)" }}>
                    ▪ {group} <span style={{ fontWeight:400, color:"#1c1c3a", marginLeft:6 }}>({items.length})</span>
                  </td>
                </tr>
                {items.map((item, i) => <Row key={item.name} item={item} idx={i}/>)}
              </>
            ))}
            {filtered.length===0 && <tr><td colSpan={5} style={{ padding:"60px", textAlign:"center", color:"var(--text-dim)" }}>No entries match. Try different search or reset filters.</td></tr>}
          </tbody>
        </table>
      </div>
      <div style={{ padding:"16px", textAlign:"center", fontSize:"0.62rem", color:"var(--bg-overlay)", borderTop:"1px solid #0d0d1e" }}>
        PyTorch · Skorch · pytorch-tabnet · PyTorch Lightning · torchvision · transformers (HuggingFace) · torch-lr-finder · timm · albumentations
      </div>
    </div></div>
  );
}