import { useState, useMemo } from "react";
import Header from "./Header";
const GROUP_ACCENT = {
  "Model Serialization":       "#60a5fa",
  "Experiment Tracking":       "#818cf8",
  "Model Registry":            "#a78bfa",
  "Model Serving & APIs":      "#34d399",
  "Containerization":          "#22d3ee",
  "CI/CD for ML":              "#4ade80",
  "Data & Model Versioning":   "#facc15",
  "Feature Stores":            "#fb923c",
  "Data Validation":           "#f472b6",
  "Production Monitoring":     "#38bdf8",
  "Drift Detection":           "#f87171",
  "Deployment Strategies":     "#c084fc",
  "Pipeline Orchestration":    "#86efac",
  "Retraining & Governance":   "#fbbf24",
  "MLOps Anti-Patterns":       "#fca5a5",
};

const STAGE_STYLE = {
  "📦 Package":    { bg:"#1e3a5f", color:"#60a5fa",  border:"#60a5fa"  },
  "🔬 Track":      { bg:"#2a1a3a", color:"#818cf8",  border:"#818cf8"  },
  "🚀 Serve":      { bg:"#1a3a2a", color:"#34d399",  border:"#34d399"  },
  "🐳 Infra":      { bg:"#1a2a3a", color:"#22d3ee",  border:"#22d3ee"  },
  "🔄 Automate":   { bg:"#2a2a1a", color:"#4ade80",  border:"#4ade80"  },
  "📊 Monitor":    { bg:"#1a3a3a", color:"#38bdf8",  border:"#38bdf8"  },
  "🔴 Anti-Pattern":{ bg:"#3a0a0a", color:"#f87171",  border:"#f87171"  },
  "🏗️ Orchestrate":{ bg:"#1a3a1a", color:"#86efac",  border:"#86efac"  },
};

const ITEMS = [
  // ═══════════════════════════════════════════
  // MODEL SERIALIZATION
  // ═══════════════════════════════════════════
  {
    name: "joblib / pickle",
    group: "Model Serialization",
    stage: "📦 Package",
    lib: "joblib · pickle",
    api: "joblib.dump(model, 'model.joblib') / joblib.load('model.joblib')",
    what: "Python-native serialization for sklearn and most Python ML objects. joblib is preferred over pickle for sklearn models because it handles large numpy arrays more efficiently via memory-mapped files.",
    when: "sklearn models, simple Python ML objects, internal tooling where the Python version is controlled and consistent. Fast to implement — the right default for sklearn pipelines.",
    pitfall: "Pickled objects are tied to the Python version, library version, and class definition at save time. A model saved with sklearn 1.2 may fail to load in sklearn 1.5. Always pin library versions when serializing. Never unpickle files from untrusted sources — it executes arbitrary code.",
    alternatives: "ONNX for cross-language/framework portability. MLflow model format for versioned registry storage.",
    code: `import joblib
from sklearn.pipeline import Pipeline

# Save the full pipeline (preprocessor + model together):
pipeline = Pipeline([('scaler', scaler), ('model', lgbm_model)])
pipeline.fit(X_train, y_train)

# Save with metadata in the filename:
joblib.dump(pipeline, 'models/lgbm_v1_cv0.8723.joblib', compress=3)

# Load and predict:
loaded_pipeline = joblib.load('models/lgbm_v1_cv0.8723.joblib')
preds = loaded_pipeline.predict_proba(X_test)[:, 1]

# Always save together: model + feature names + threshold
import json
metadata = {
    'features': list(X_train.columns),
    'threshold': 0.42,
    'cv_auc': 0.8723,
    'train_date': '2025-01-15',
}
with open('models/lgbm_v1_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)`,
    links: "Model Registry · sklearn Pipeline · Experiment Tracking",
  },
  {
    name: "ONNX (Open Neural Network Exchange)",
    group: "Model Serialization",
    stage: "📦 Package",
    lib: "onnx · onnxruntime · skl2onnx · torch.onnx",
    api: "torch.onnx.export(model, dummy_input, 'model.onnx') / skl2onnx.convert_sklearn(model)",
    what: "Open standard format for ML models that enables cross-framework and cross-language inference. An ONNX model can be trained in PyTorch, exported to ONNX, and then deployed in C++, Java, or .NET using the ONNX Runtime. Provides 2-5× inference speedup over Python via graph optimization.",
    when: "Production deployment where inference speed is critical. Cross-language deployments (model trained in Python, serving in C++/Java/.NET). Edge deployment (ONNX Runtime for mobile/embedded). Any serious production system should export to ONNX.",
    pitfall: "Not all PyTorch operations are ONNX-compatible — dynamic control flow (if/else based on tensor values) requires `torch.jit.script` treatment or operator workarounds. Always validate ONNX output against PyTorch output numerically after export.",
    alternatives: "TorchScript for PyTorch-only production. TensorFlow SavedModel for TF/Keras. TFLite/Core ML for on-device.",
    code: `import torch
import onnx
import onnxruntime as ort
import numpy as np

# Export PyTorch model to ONNX:
model.eval()
dummy_input = torch.randn(1, input_size)  # Batch of 1 with correct feature dim

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=17,             # Use latest stable opset
    input_names=['features'],
    output_names=['logits'],
    dynamic_axes={                # Enable variable batch size
        'features': {0: 'batch_size'},
        'logits':   {0: 'batch_size'},
    }
)

# Validate: check ONNX model is valid
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# Run inference with ONNX Runtime (no PyTorch needed):
session = ort.InferenceSession('model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

X_np = X_test.astype(np.float32)
outputs = session.run(None, {'features': X_np})
onnx_logits = outputs[0]

# Verify numerical equivalence with PyTorch output:
with torch.no_grad():
    torch_logits = model(torch.tensor(X_np)).numpy()
np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-4)
print("ONNX and PyTorch outputs match ✓")

# For sklearn models:
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onnx_sklearn = convert_sklearn(sklearn_pipeline,
    initial_types=[('input', FloatTensorType([None, n_features]))])
with open('sklearn_model.onnx', 'wb') as f:
    f.write(onnx_sklearn.SerializeToString())`,
    links: "Model Serving & APIs · Containerization · Deployment Strategies",
  },
  {
    name: "TorchScript",
    group: "Model Serialization",
    stage: "📦 Package",
    lib: "torch.jit",
    api: "torch.jit.trace(model, example) / torch.jit.script(model)",
    what: "Serializes a PyTorch model into a static computation graph that can be loaded and run without Python. Two modes: `trace` follows the execution path of one example (fast but misses dynamic branches); `script` statically analyzes the full Python code (handles conditionals).",
    when: "PyTorch-to-C++ deployment (LibTorch). Embedding a PyTorch model in a mobile app (PyTorch Mobile). Anywhere you need to run PyTorch inference without a Python interpreter.",
    pitfall: "trace() silently ignores Python control flow — if your model has `if x.shape[0] > 1:` branches, trace() captures only the branch taken by the example input. Use script() for models with dynamic control flow.",
    alternatives: "ONNX for cross-framework. For pure Python serving, saving state_dict and reconstructing in Python is simpler.",
    code: `import torch

model.eval()

# Tracing (fast, no dynamic control flow):
example_input  = torch.randn(1, input_size)
traced_model   = torch.jit.trace(model, example_input)
traced_model.save('traced_model.pt')

# Scripting (full Python support, handles branches):
scripted_model = torch.jit.script(model)
scripted_model.save('scripted_model.pt')

# Load without Python class definition (great for C++ serving):
loaded = torch.jit.load('scripted_model.pt')
loaded.eval()
with torch.no_grad():
    output = loaded(example_input)

# Test equivalence before deploying:
with torch.no_grad():
    orig   = model(example_input)
    script = scripted_model(example_input)
torch.testing.assert_close(orig, script)
print("TorchScript matches original ✓")`,
    links: "Model Serving & APIs · ONNX · Containerization",
  },

  // ═══════════════════════════════════════════
  // EXPERIMENT TRACKING
  // ═══════════════════════════════════════════
  {
    name: "MLflow Tracking",
    group: "Experiment Tracking",
    stage: "🔬 Track",
    lib: "mlflow",
    api: "mlflow.start_run() / mlflow.log_params() / mlflow.log_metrics() / mlflow.sklearn.log_model()",
    what: "Open-source platform for the complete ML lifecycle. The Tracking component logs parameters, metrics, artifacts, and code version for every experiment run. Provides a UI to compare runs, visualize metrics over time, and reproduce any historical experiment.",
    when: "Any ML project beyond one-off notebooks. Every training run should be tracked. When you need to compare 50 hyperparameter tuning runs and find the best one a week later. Industry standard — use it from day one.",
    pitfall: "Forgetting to call mlflow.end_run() leaves runs in a `RUNNING` state indefinitely. Use the context manager (`with mlflow.start_run():`) to guarantee cleanup. Also: log the full pipeline artifact, not just the model weights, so the pipeline can be fully reproduced.",
    alternatives: "Weights & Biases (richer visualizations, better team collaboration), Neptune.ai (enterprise features), ClearML (self-hosted).",
    code: `import mlflow
import mlflow.sklearn
import mlflow.lightgbm

mlflow.set_experiment("credit_default_prediction")

with mlflow.start_run(run_name="lgbm_baseline_v1"):
    # Log hyperparameters:
    mlflow.log_params({
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "cv_strategy": "StratifiedKFold(5)",
        "feature_set": "v3_with_ratio_features",
    })

    # Train model (your training code here)
    model = train_lgbm(X_train, y_train)

    # Log metrics:
    mlflow.log_metrics({
        "oof_auc": oof_auc,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "train_time_sec": train_time,
    })

    # Log the full sklearn pipeline as a reproducible artifact:
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="credit_default_lgbm",  # Registers in Model Registry
        input_example=X_train[:5],                    # Documents expected input format
        signature=mlflow.models.infer_signature(X_train, pipeline.predict(X_train)),
    )

    # Log any other artifacts (feature importance plot, confusion matrix, etc.):
    mlflow.log_artifact("reports/feature_importance.png")
    mlflow.log_artifact("data/feature_list_v3.txt")

    # Log dataset metadata:
    mlflow.set_tags({
        "train_rows": len(X_train),
        "n_features": X_train.shape[1],
        "git_commit": "abc1234",
    })

print(f"Run ID: {mlflow.active_run().info.run_id}")`,
    links: "Model Registry · CI/CD for ML · Data & Model Versioning",
  },
  {
    name: "Weights & Biases (W&B)",
    group: "Experiment Tracking",
    stage: "🔬 Track",
    lib: "wandb",
    api: "wandb.init() / wandb.log() / wandb.Artifact() / wandb.sweep()",
    what: "Cloud-based experiment tracking with richer visualization than MLflow: interactive loss curves, confusion matrices, sample image grids, system metrics (GPU utilization, memory), and built-in hyperparameter sweep integration. Strong team collaboration features.",
    when: "Deep learning projects where you want visual monitoring of training curves, GPU utilization, and sample predictions. Team environments where multiple people run experiments simultaneously. When MLflow's local UI is insufficient for your reporting needs.",
    pitfall: "W&B sends data to Weights & Biases cloud by default — sensitive data should use W&B on-premises or MLflow locally. wandb.log() inside the training loop per-step creates very large run histories for long training — log per epoch to manage data volume.",
    alternatives: "MLflow (local, open-source, sklearn-centric), Neptune.ai (enterprise), ClearML (self-hosted).",
    code: `import wandb

# Initialize a run:
run = wandb.init(
    project="tabular_classification",
    name="lgbm_baseline",
    config={                         # Hyperparameters auto-tracked
        "n_estimators": 500,
        "learning_rate": 0.05,
        "cv_folds": 5,
    },
    tags=["lgbm", "baseline", "v1"],
)

# Log metrics at each fold:
for fold, (score, features) in enumerate(cv_results):
    wandb.log({
        "fold": fold,
        "val_auc": score,
        "n_features": len(features),
    })

# Log the final model as a versioned artifact:
artifact = wandb.Artifact("credit_model", type="model",
    metadata={"oof_auc": oof_auc, "threshold": 0.42})
artifact.add_file("models/lgbm_v1.joblib")
run.log_artifact(artifact)

# Hyperparameter sweep (Bayesian search):
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_auc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.005, "max": 0.3},
        "num_leaves":    {"values": [31, 63, 127]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="tabular_classification")
wandb.agent(sweep_id, function=train_and_log, count=50)

wandb.finish()`,
    links: "Hyperparameter Tuning · Deep Learning Training · CI/CD for ML",
  },

  // ═══════════════════════════════════════════
  // MODEL REGISTRY
  // ═══════════════════════════════════════════
  {
    name: "MLflow Model Registry",
    group: "Model Registry",
    stage: "📦 Package",
    lib: "mlflow",
    api: "mlflow.register_model() / client.transition_model_version_stage()",
    what: "Centralised repository for managing model versions with lifecycle stages: Staging, Production, Archived. Provides lineage (which run produced this model), approvals workflow, and a single source of truth for the 'current production model'. Prevents the chaos of manually-tracked model files.",
    when: "Any team with more than one data scientist deploying models. When you need to know 'which version is in production right now and what run produced it'. When model rollbacks need to be auditable.",
    pitfall: "The Model Registry stores a reference to the artifact, not the artifact itself — the underlying MLflow artifact store must be accessible at deployment time. In production, use a shared artifact store (S3, GCS, Azure Blob) so all services can access model artifacts.",
    alternatives: "W&B Artifact Registry, Neptune model registry, custom model store in S3 with a DynamoDB lookup table.",
    code: `from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

# Register a model (can also be done inside mlflow.sklearn.log_model):
model_uri  = f"runs:/{run_id}/model"
model_name = "credit_default_lgbm"
result = mlflow.register_model(model_uri, model_name)
print(f"Registered version: {result.version}")

# Transition to Staging after validation:
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging",
    archive_existing_versions=False,  # Keep old staging models
)

# After QA passes, promote to Production:
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production",
    archive_existing_versions=True,   # Archive previous production model
)

# Load the current production model in serving code:
production_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)
predictions = production_model.predict(X_new)

# Rollback: promote previous version back to Production
client.transition_model_version_stage(
    name=model_name,
    version=str(int(result.version) - 1),
    stage="Production",
    archive_existing_versions=True,
)`,
    links: "MLflow Tracking · Model Serving & APIs · Deployment Strategies",
  },

  // ═══════════════════════════════════════════
  // MODEL SERVING & APIs
  // ═══════════════════════════════════════════
  {
    name: "FastAPI Model Server",
    group: "Model Serving & APIs",
    stage: "🚀 Serve",
    lib: "fastapi · uvicorn · pydantic",
    api: "FastAPI app → uvicorn main:app --host 0.0.0.0 --port 8000",
    what: "High-performance Python web framework for building ML inference APIs. Provides automatic request validation via Pydantic schemas, OpenAPI documentation out of the box, async support for concurrent requests, and clean error handling. The most commonly used framework for custom ML APIs in 2025.",
    when: "Any ML model that needs to serve predictions over HTTP. Custom preprocessing/postprocessing logic that BentoML or MLflow serving can't handle. Microservice architectures. The right default for most ML serving tasks.",
    pitfall: "FastAPI is single-process by default — a blocking model prediction will block all concurrent requests. Use async handlers for non-blocking I/O, or run with multiple workers: `uvicorn main:app --workers 4`. For CPU-bound inference, use ProcessPoolExecutor to avoid GIL blocking.",
    alternatives: "Flask (simpler, less performant), BentoML (more ML-specific, handles batching), TorchServe (PyTorch-specific), Triton Inference Server (GPU-optimised, multi-model).",
    code: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import joblib
import json
from typing import List

# Load model and metadata once at startup:
pipeline = joblib.load("models/lgbm_v1.joblib")
with open("models/lgbm_v1_metadata.json") as f:
    metadata = json.load(f)

app = FastAPI(title="Credit Default Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]                   # Input feature vector

    @validator('features')
    def check_length(cls, v):
        expected = len(metadata['features'])
        if len(v) != expected:
            raise ValueError(f"Expected {expected} features, got {len(v)}")
        return v

class PredictionResponse(BaseModel):
    probability: float
    prediction: int                          # 0 or 1 at chosen threshold
    threshold: float
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata.get("version", "v1")}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        X = np.array([request.features], dtype=np.float32)
        prob = float(pipeline.predict_proba(X)[0, 1])
        threshold = metadata.get("threshold", 0.5)
        return PredictionResponse(
            probability=round(prob, 4),
            prediction=int(prob >= threshold),
            threshold=threshold,
            model_version=metadata.get("version", "v1"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: List[PredictionRequest]):
    X = np.array([r.features for r in requests], dtype=np.float32)
    probs = pipeline.predict_proba(X)[:, 1]
    threshold = metadata.get("threshold", 0.5)
    return [{"probability": round(float(p), 4), "prediction": int(p >= threshold)}
            for p in probs]

# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4`,
    links: "Containerization · Deployment Strategies · Data Validation · Production Monitoring",
  },
  {
    name: "BentoML",
    group: "Model Serving & APIs",
    stage: "🚀 Serve",
    lib: "bentoml",
    api: "bentoml.sklearn.save_model() / @bentoml.service / bentoml serve",
    what: "ML-specific serving framework that handles model packaging, versioning, adaptive batching, and deployment to cloud platforms with minimal boilerplate. Particularly strong for batching — automatically groups simultaneous requests into a single batch for efficient GPU utilisation.",
    when: "When you need automatic request batching for neural network inference (critical for GPU utilisation). Multi-model pipelines where one request invokes several models. When you want one-command deployment to AWS Lambda, ECS, or Kubernetes.",
    pitfall: "BentoML's Runner abstraction handles batching but requires careful configuration of max_batch_size and max_latency_ms. Setting max_latency_ms too low defeats batching; too high increases tail latency. Profile your throughput requirements first.",
    alternatives: "FastAPI (more flexibility), Triton (better for multi-GPU), TorchServe (PyTorch-specific), MLflow serving (simplest for sklearn).",
    code: `import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, JSON

# Save model to BentoML model store:
bentoml.sklearn.save_model(
    "credit_model",
    pipeline,
    signatures={"predict_proba": {"batchable": True, "batch_dim": 0}},
    metadata={"cv_auc": 0.8723, "threshold": 0.42},
)

# Define the service:
credit_runner = bentoml.sklearn.get("credit_model:latest").to_runner()

svc = bentoml.Service("credit_prediction", runners=[credit_runner])

@svc.api(input=NumpyNdarray(dtype=np.float32, shape=(-1, N_FEATURES)),
         output=JSON())
async def predict(input_data: np.ndarray):
    probs = await credit_runner.predict_proba.async_run(input_data)
    threshold = 0.42
    return {
        "probabilities": probs[:, 1].tolist(),
        "predictions":   (probs[:, 1] >= threshold).astype(int).tolist(),
    }

# Serve locally: bentoml serve service:svc --reload
# Build container: bentoml containerize credit_prediction:latest
# Deploy to cloud: bentoml deployment create credit_prediction --bento credit_prediction:latest`,
    links: "Containerization · Production Monitoring · Model Registry",
  },
  {
    name: "MLflow Serving",
    group: "Model Serving & APIs",
    stage: "🚀 Serve",
    lib: "mlflow",
    api: "mlflow models serve -m models:/model_name/Production -p 5000",
    what: "One-command REST API server for any MLflow-logged model. Zero boilerplate — provide the model URI and MLflow starts a Flask server exposing /invocations and /health endpoints. Handles input/output schema validation via the model signature.",
    when: "Rapid prototyping and internal tooling where simplicity matters more than performance. Staging/QA environments where you want to quickly serve a registered model for testing. Not recommended for high-throughput production (use FastAPI or BentoML instead).",
    pitfall: "MLflow serving uses Flask single-threaded by default — not suitable for concurrent production traffic. Use --enable-mlserver flag to use MLServer as a backend for better performance, or migrate to a proper serving framework for production.",
    alternatives: "FastAPI for production. BentoML for batching. For cloud: SageMaker, Vertex AI, or Azure ML managed endpoints.",
    code: `# Serve from Model Registry (one command):
# mlflow models serve -m "models:/credit_default_lgbm/Production" -p 5000 --no-conda

# Call the endpoint:
import requests, json
import pandas as pd

features = X_test.iloc[:5].to_dict(orient='records')
response = requests.post(
    "http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"dataframe_records": features}),
)
predictions = response.json()

# Build a Docker image from an MLflow model (for deployment):
# mlflow models build-docker -m "models:/credit_default_lgbm/Production" -n "credit-model-image"

# Or generate a Dockerfile for inspection:
# mlflow models generate-dockerfile -m "models:/credit_default_lgbm/Production" --output-directory ./docker_model`,
    links: "Model Registry · Containerization · FastAPI Model Server",
  },
  {
    name: "Triton Inference Server",
    group: "Model Serving & APIs",
    stage: "🚀 Serve",
    lib: "tritonclient · tritonserver (NVIDIA)",
    api: "tritonserver --model-repository=/models / tritonclient.http.InferenceServerClient",
    what: "NVIDIA's high-performance inference server for GPU-accelerated model serving. Supports ONNX, TorchScript, TensorFlow, TensorRT, and Python backends simultaneously. Features dynamic batching, model ensemble pipelines, and concurrent model execution on GPU. The industry standard for high-throughput DL serving.",
    when: "High-throughput production ML systems with GPU inference. Multiple models sharing a GPU (Triton manages GPU memory allocation). When you need <10ms p99 latency at 1000+ RPS. Computer vision and NLP at scale.",
    pitfall: "Triton requires model repository structure to follow strict naming conventions. The config.pbtxt file must exactly specify input/output tensor names, shapes, and dtypes. A single wrong dimension in config.pbtxt crashes the model load silently.",
    alternatives: "FastAPI + ONNX Runtime for simpler GPU serving. TorchServe for PyTorch-only. BentoML Runners for managed batching.",
    code: `# Model repository structure Triton expects:
# /models/
#   credit_model/
#     config.pbtxt      ← Model configuration
#     1/                ← Version 1
#       model.onnx      ← ONNX model file

# config.pbtxt:
"""
name: "credit_model"
backend: "onnxruntime"
max_batch_size: 256
input [{
  name: "features"
  data_type: TYPE_FP32
  dims: [20]
}]
output [{
  name: "logits"
  data_type: TYPE_FP32
  dims: [2]
}]
dynamic_batching {
  preferred_batch_size: [32, 64, 128]
  max_queue_delay_microseconds: 5000
}
"""

# Client call:
import tritonclient.http as tritonhttpclient
import numpy as np

client = tritonhttpclient.InferenceServerClient(url="localhost:8000")

inputs  = [tritonhttpclient.InferInput("features", [1, 20], "FP32")]
outputs = [tritonhttpclient.InferRequestedOutput("logits")]
inputs[0].set_data_from_numpy(X_test[:1].astype(np.float32))

result  = client.infer(model_name="credit_model", inputs=inputs, outputs=outputs)
logits  = result.as_numpy("logits")`,
    links: "ONNX · Containerization · Production Monitoring",
  },

  // ═══════════════════════════════════════════
  // CONTAINERIZATION
  // ═══════════════════════════════════════════
  {
    name: "Docker for ML",
    group: "Containerization",
    stage: "🐳 Infra",
    lib: "docker",
    api: "docker build -t ml-api:v1 . / docker run -p 8000:8000 ml-api:v1",
    what: "Container technology that packages an application and all its dependencies into an isolated, reproducible image. For ML, Docker solves the 'it works on my machine' problem — the model, Python version, library versions, and system dependencies are all frozen. The foundation of every production ML deployment.",
    when: "Every production ML deployment. Before any cloud deployment. When library version mismatches cause inference failures. Docker is non-negotiable for production — it's how you guarantee reproducibility between training, staging, and production environments.",
    pitfall: "Installing dependencies at container runtime (`pip install` in CMD) defeats the purpose — always install all dependencies during the build step (RUN pip install). Use multi-stage builds to keep the final image small. Pin all library versions explicitly — `torch==2.3.0` not `torch`.",
    alternatives: "Podman (Docker-compatible, rootless), conda-pack (packages conda environment without Docker), uv (ultra-fast Python packaging).",
    code: `# Dockerfile for FastAPI ML serving:
FROM python:3.11-slim as base

# Install system dependencies (separate layer — cached unless changed):
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 \\                     # Required by LightGBM
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (separate layer — cached unless requirements.txt changes):
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model (changes most frequently — last layer):
COPY src/ ./src/
COPY models/ ./models/

# Non-root user for security:
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# requirements.txt — always pin versions:
# fastapi==0.111.0
# uvicorn[standard]==0.29.0
# scikit-learn==1.5.0
# lightgbm==4.3.0
# pydantic==2.7.0
# numpy==1.26.4

# Build and run:
# docker build -t credit-api:v1.2.0 .
# docker run -p 8000:8000 --memory="2g" --cpus="2" credit-api:v1.2.0`,
    links: "CI/CD for ML · Deployment Strategies · Model Serving & APIs",
  },
  {
    name: "Kubernetes for ML Serving",
    group: "Containerization",
    stage: "🐳 Infra",
    lib: "kubectl · helm · k8s manifests",
    api: "kubectl apply -f deployment.yaml / kubectl rollout undo deployment/ml-api",
    what: "Container orchestration platform that manages deployment, scaling, and rollback of containerised ML services. Key ML capabilities: horizontal pod autoscaling (scale on CPU/GPU utilisation), rolling updates (zero-downtime model updates), and easy A/B testing via traffic splitting between model versions.",
    when: "Production systems that need automatic scaling, self-healing (restart failed pods), and zero-downtime deployments. When multiple ML services share infrastructure. Any system serving > 100 RPM consistently.",
    pitfall: "Kubernetes has a steep learning curve. For small teams, managed Kubernetes (EKS, GKE, AKS) significantly reduces operational overhead. Don't manage your own Kubernetes cluster unless you have dedicated DevOps/platform engineers.",
    alternatives: "AWS ECS (simpler than Kubernetes, AWS-only), Cloud Run/Cloud Functions (serverless, auto-scales to zero), Railway/Render (simple managed hosting for lower traffic).",
    code: `# deployment.yaml — Kubernetes deployment for ML API:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-model-api
  labels: {app: credit-model}
spec:
  replicas: 3
  selector:
    matchLabels: {app: credit-model}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # Max extra pods during update
      maxUnavailable: 0   # Zero downtime
  template:
    metadata:
      labels: {app: credit-model, version: v1-2-0}
    spec:
      containers:
      - name: credit-api
        image: registry.io/credit-api:v1.2.0
        ports: [{containerPort: 8000}]
        resources:
          requests: {memory: "512Mi", cpu: "500m"}
          limits:   {memory: "2Gi",  cpu: "2"}
        livenessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 30
          periodSeconds: 10
        env:
        - name: MODEL_VERSION
          value: "v1.2.0"

---
# Horizontal Pod Autoscaler:
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: credit-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: credit-model-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: {type: Utilization, averageUtilization: 70}`,
    links: "Docker for ML · Deployment Strategies · CI/CD for ML",
  },

  // ═══════════════════════════════════════════
  // CI/CD FOR ML
  // ═══════════════════════════════════════════
  {
    name: "GitHub Actions for ML CI/CD",
    group: "CI/CD for ML",
    stage: "🔄 Automate",
    lib: "GitHub Actions · actions/checkout · pytest · DVC",
    api: ".github/workflows/ml_pipeline.yml",
    what: "Automates the ML pipeline on every code push: run unit tests on preprocessing/feature code, validate data schemas, retrain the model (or verify performance on cached data), evaluate against production metrics, and optionally deploy if evaluation passes. Brings software engineering discipline to ML.",
    when: "Any ML project where more than one person contributes code. When retraining happens regularly. When you want to prevent a bad feature change from silently degrading production performance. CI/CD is the operationalisation of your ML practice.",
    pitfall: "Training the full model in CI on every push is too slow for large models. Use a fast smoke test (train on 1% of data) in CI for pre-merge checks, and reserve full retraining for a scheduled nightly/weekly job. Cache model artifacts with DVC to avoid repeated downloads.",
    alternatives: "GitLab CI (self-hosted), Azure DevOps Pipelines, Kubeflow Pipelines (Kubernetes-native), Prefect or Airflow for the training part.",
    code: `# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Lint and type check
      run: |
        ruff check src/
        mypy src/

    - name: Run unit tests (feature engineering, preprocessing)
      run: pytest tests/ -v --tb=short

    - name: Validate data schema with Great Expectations
      run: python scripts/validate_data.py --suite training_data_suite

    - name: Smoke test retraining (1% of data)
      run: python scripts/train.py --sample-fraction 0.01 --smoke-test

  deploy:
    needs: test-and-validate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - name: Build and push Docker image
      run: |
        docker build -t $IMAGE_TAG .
        docker push $IMAGE_TAG

    - name: Deploy to staging
      run: kubectl set image deployment/credit-model credit-api=$IMAGE_TAG

    - name: Run integration tests against staging
      run: pytest tests/integration/ --base-url=https://staging.api.example.com`,
    links: "Docker for ML · Model Registry · Data Validation · MLflow Tracking",
  },
  {
    name: "DVC (Data Version Control)",
    group: "Data & Model Versioning",
    stage: "🔄 Automate",
    lib: "dvc",
    api: "dvc init / dvc add data/ / dvc push / dvc pull / dvc repro",
    what: "Git extension for versioning large data files and ML pipelines. Stores data in remote storage (S3, GCS, local) while tracking it with lightweight pointer files in Git. Enables `git checkout experiment_v2 && dvc pull` to reproduce any historical experiment with its exact data snapshot.",
    when: "Any project where datasets evolve over time. When multiple experiments use different versions of the same dataset. When you need to reproduce an exact model from 6 months ago — DVC makes this possible by pairing the code commit with its data version.",
    pitfall: "DVC is not a replacement for a proper data platform. For very large datasets (> 100GB), DVC pull is slow — use a cloud feature store or data versioning platform (LakeFS, Delta Lake) instead. DVC works best for datasets that change infrequently (weekly/monthly).",
    alternatives: "LakeFS (Git-like semantics for data lakes), Delta Lake (ACID transactions + versioning for Spark/data warehouses), Pachyderm (Kubernetes-native data versioning).",
    code: `# Initial setup in a project:
git init
dvc init          # Creates .dvc/ directory and .gitignore entries

# Add a dataset to DVC tracking:
dvc add data/raw/train.csv                  # Creates data/raw/train.csv.dvc
git add data/raw/train.csv.dvc .gitignore
git commit -m "Add raw training dataset v1"

# Configure remote storage (S3 example):
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push                                    # Upload data to S3

# On another machine / CI environment:
git clone <repo>
dvc pull                                    # Download exact data version

# Define a reproducible pipeline (dvc.yaml):
# stages:
#   prepare:
#     cmd: python src/prepare.py
#     deps: [src/prepare.py, data/raw/train.csv]
#     outs: [data/processed/train_features.parquet]
#   train:
#     cmd: python src/train.py
#     deps: [src/train.py, data/processed/train_features.parquet]
#     outs: [models/model.joblib]
#     metrics: [reports/metrics.json]

dvc repro          # Re-runs only stages with changed deps
dvc params diff    # Compare params between commits
dvc metrics show   # Display all tracked metrics`,
    links: "CI/CD for ML · MLflow Tracking · Data Validation",
  },

  // ═══════════════════════════════════════════
  // FEATURE STORES
  // ═══════════════════════════════════════════
  {
    name: "Feature Store Concepts & Why They Matter",
    group: "Feature Stores",
    stage: "📦 Package",
    lib: "feast · tecton · hopsworks",
    api: "store.get_historical_features() / store.get_online_features()",
    what: "A system that stores, serves, and manages features consistently across training and inference. Solves training-serving skew — the most common cause of silent model failures in production. The feature store computes features once, stores them, and serves the same values to both the training pipeline and the online inference server.",
    when: "Any production ML system where features are computed from raw data before serving. When the same features are used across multiple models. When training-serving skew is suspected (model CV looks good but production performance is poor). Enterprise ML teams.",
    pitfall: "Training-serving skew occurs when feature computation in training (batch, historical) differs from inference (real-time, streaming). A feature store eliminates this by enforcing a single definition. Even without a dedicated feature store, always version and test your feature transformation code as carefully as your model code.",
    alternatives: "Manual feature engineering with strict pipeline versioning (acceptable for small teams), Redis for simple online feature serving, Snowflake ML feature store (for Snowflake data warehouse users).",
    code: `# Feast feature store — minimal example:
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo/")

# For training: fetch historical features joined to entity table
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003],
    "event_timestamp": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"]),
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_stats:total_spend_30d",
        "customer_stats:n_transactions_30d",
        "customer_stats:avg_transaction_value",
        "customer_risk:credit_score",
    ],
).to_df()

# For inference: fetch real-time online features
online_features = store.get_online_features(
    features=[
        "customer_stats:total_spend_30d",
        "customer_stats:n_transactions_30d",
    ],
    entity_rows=[{"customer_id": 1001}],  # Real-time lookup
).to_df()

# Guarantee: training_df and online_features use IDENTICAL feature computation
# → eliminates training-serving skew`,
    links: "Data Validation · Production Monitoring · Drift Detection",
  },

  // ═══════════════════════════════════════════
  // DATA VALIDATION
  // ═══════════════════════════════════════════
  {
    name: "Great Expectations",
    group: "Data Validation",
    stage: "🔄 Automate",
    lib: "great_expectations",
    api: "context.run_checkpoint(checkpoint_name='my_checkpoint')",
    what: "Data quality framework that lets you define Expectations (assertions about your data) and validate them automatically at every pipeline run. An expectation might be: 'this column has no nulls', 'this column is between 0 and 1', 'the cardinality of this column is between 50 and 200'. Generates HTML data quality reports.",
    when: "Any production ML pipeline where data quality issues would cause silent model failure. Data ingestion pipelines. Before every training run to ensure data hasn't changed in unexpected ways. CI/CD validation gates.",
    pitfall: "Great Expectations has significant setup overhead. For quick validation in a simpler project, Pandera (declarative schema for DataFrames) or simple assert statements are faster to implement. Only invest in Great Expectations if you have recurring automated pipelines.",
    alternatives: "Pandera (simpler, decorator-based), Pydantic (for JSON/dict validation at API boundaries), TFDV (TensorFlow Data Validation — strong for distribution comparison), Soda Core.",
    code: `import great_expectations as gx
import pandas as pd

# Create expectation suite for training data:
context  = gx.get_context()
suite    = context.add_expectation_suite("training_data_suite")

validator = context.get_validator(
    batch_request=...,
    expectation_suite_name="training_data_suite"
)

# Define expectations:
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("target")
validator.expect_column_values_to_be_between("age", min_value=18, max_value=100)
validator.expect_column_values_to_be_in_set("target", [0, 1])
validator.expect_table_row_count_to_be_between(min_value=10000, max_value=5000000)
validator.expect_column_unique_value_count_to_be_between("customer_id", min_value=5000)
validator.expect_column_mean_to_be_between("income", min_value=20000, max_value=200000)

# Catch distribution shifts:
validator.expect_column_kl_divergence_to_be_less_than("income",
    partition_object=reference_partition, threshold=0.1)

validator.save_expectation_suite()

# Run in CI/CD pipeline:
results = context.run_checkpoint("training_checkpoint")
if not results["success"]:
    raise ValueError("Data validation FAILED — aborting pipeline")

print(results.to_json_dict())`,
    links: "CI/CD for ML · Production Monitoring · Drift Detection · Feature Stores",
  },
  {
    name: "Pandera — DataFrame Schema Validation",
    group: "Data Validation",
    stage: "🔄 Automate",
    lib: "pandera",
    api: "@pa.check_input(schema) / @pa.check_output(schema) / schema.validate(df)",
    what: "Lightweight, Pythonic data validation library for pandas DataFrames. Define a schema declaratively (column types, ranges, regex patterns, custom checks) and validate DataFrames at any point in the pipeline. Extremely easy to integrate into existing code via decorators.",
    when: "Any function that accepts or returns a DataFrame. API boundaries where DataFrames are constructed from user input. Preprocessing functions where wrong input would produce wrong output silently. Much simpler to adopt than Great Expectations for Python-centric teams.",
    pitfall: "Pandera validates DataFrame structure and column-level statistics but is not designed for complex cross-table or temporal data quality checks — use Great Expectations for those. Validation adds latency — apply it at pipeline boundaries, not inside tight loops.",
    alternatives: "Great Expectations (richer reporting, more enterprise features), pydantic (for dict/JSON schemas), marshmallow (serialisation + validation).",
    code: `import pandera as pa
from pandera.typing import DataFrame, Series

# Define schema:
class TrainingDataSchema(pa.SchemaModel):
    customer_id:   Series[int]    = pa.Field(unique=True, gt=0)
    age:           Series[float]  = pa.Field(ge=18, le=120, nullable=False)
    income:        Series[float]  = pa.Field(ge=0, le=1e7, nullable=True)
    debt_ratio:    Series[float]  = pa.Field(ge=0, le=10)
    target:        Series[int]    = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        coerce     = True          # Auto-cast column types
        strict     = "filter"      # Remove unexpected columns silently

# Validate a DataFrame:
try:
    validated_df = TrainingDataSchema.validate(raw_df, lazy=True)  # lazy=True collects all errors
except pa.errors.SchemaErrors as e:
    print("Schema validation failed:")
    print(e.failure_cases)        # DataFrame of all failed checks

# Use as a decorator:
@pa.check_input(TrainingDataSchema)
@pa.check_output(TrainingDataSchema)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Any DataFrame entering or leaving this function is validated
    return df.dropna(subset=['target'])`,
    links: "CI/CD for ML · Data Validation · Production Monitoring",
  },

  // ═══════════════════════════════════════════
  // PRODUCTION MONITORING
  // ═══════════════════════════════════════════
  {
    name: "Evidently AI",
    group: "Production Monitoring",
    stage: "📊 Monitor",
    lib: "evidently",
    api: "Report(metrics=[DataDriftPreset()]).run(reference_data=..., current_data=...)",
    what: "Open-source ML monitoring library that generates comprehensive reports on data drift, model performance, and data quality. Provides HTML/JSON reports for offline analysis and a real-time dashboard (Evidently Cloud or self-hosted). The most accessible production monitoring tool for data scientists.",
    when: "Any production model that needs to be monitored for drift. Batch scoring pipelines where you can compare each batch to the training distribution. Weekly/monthly monitoring reports for stakeholders. The starting point for any monitoring implementation.",
    pitfall: "Evidently requires ground truth labels for performance monitoring. For low-latency, high-frequency predictions where labels arrive late (weeks/months later), use data drift monitoring as a proxy for performance degradation.",
    alternatives: "NannyML (strong for estimating performance without labels), WhyLogs + WhyLabs (lightweight logging), Arize (enterprise, excellent UI), Fiddler (regulatory compliance focus).",
    code: `from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import *
import pandas as pd

# Reference = training data; Current = recent production data
reference_data = pd.read_parquet("data/train_features.parquet")
current_data   = pd.read_parquet("data/production_2025_01.parquet")

# Data drift report:
data_drift_report = Report(metrics=[
    DataDriftPreset(),          # Detects drift in all features
    DataQualityPreset(),        # Null rates, value ranges, cardinality
])
data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
)
data_drift_report.save_html("reports/drift_jan_2025.html")

# Model performance report (requires ground truth labels):
model_report = Report(metrics=[
    ClassificationPreset(),
    ColumnDriftMetric(column_name="prediction"),
])
model_report.run(
    reference_data=reference_with_labels,
    current_data=current_with_labels,
    column_mapping=ColumnMapping(target="target", prediction="prediction"),
)

# Programmatic drift detection:
report_dict = data_drift_report.as_dict()
drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
n_drifted      = report_dict["metrics"][0]["result"]["number_of_drifted_columns"]
print(f"Dataset drift: {drift_detected}, Drifted columns: {n_drifted}")

if drift_detected:
    send_alert(f"Data drift detected: {n_drifted} columns drifted in production batch")`,
    links: "Drift Detection · Retraining & Governance · CI/CD for ML",
  },
  {
    name: "NannyML — Performance Without Labels",
    group: "Production Monitoring",
    stage: "📊 Monitor",
    lib: "nannyml",
    api: "CBPE(y_pred_proba='proba', y_true='target').fit(reference).estimate(production)",
    what: "ML monitoring library that estimates model performance on unlabelled production data using Confidence-Based Performance Estimation (CBPE). Detects performance drops before ground truth labels arrive — critical for use cases where labels are delayed (credit default, churn, fraud). Also provides univariate drift detection.",
    when: "Any ML system where ground truth labels are delayed or unavailable in real time. Fraud detection (labels arrive weeks later when chargebacks occur). Credit scoring (default labels arrive months later). Any business-critical model where you can't wait for labels to monitor degradation.",
    pitfall: "CBPE assumes the model's confidence is well-calibrated. If your model is poorly calibrated (Random Forest, for example), CBPE estimates will be unreliable. Always calibrate your model with CalibratedClassifierCV before deploying with NannyML monitoring.",
    alternatives: "Evidently (requires labels for performance monitoring), WhyLogs (simpler, less sophisticated estimation), Arize (enterprise, better UI).",
    code: `import nannyml as nml
import pandas as pd

# CBPE estimates performance WITHOUT labels:
estimator = nml.CBPE(
    y_pred_proba  = 'predicted_proba',      # Column name of model probability output
    y_true        = 'target',               # Column name of ground truth (reference only)
    metrics       = ['roc_auc', 'f1'],
    chunk_size    = 1000,                   # Evaluate performance every 1000 predictions
    problem_type  = 'binary_classification',
)

# Fit on reference (labelled training/validation data):
estimator.fit(reference_data)

# Estimate on production (NO labels required):
estimated_performance = estimator.estimate(production_data)

# Plot performance over time:
figure = estimated_performance.plot(kind='performance', metric='roc_auc')
figure.show()

# Alert when estimated performance drops:
for chunk in estimated_performance.data.itertuples():
    if chunk.roc_auc < 0.75:  # Your performance threshold
        print(f"⚠️  Performance alert: estimated AUC={chunk.roc_auc:.3f} in chunk {chunk.key}")

# Univariate drift detection (works on any data):
univariate_drift = nml.UnivariateDriftCalculator(
    column_names=feature_columns,
    chunk_size=1000,
).fit(reference_data).calculate(production_data)`,
    links: "Drift Detection · Retraining & Governance · Poor Probability Calibration",
  },
  {
    name: "Prometheus + Grafana for ML Metrics",
    group: "Production Monitoring",
    stage: "📊 Monitor",
    lib: "prometheus_client · grafana",
    api: "Counter() / Histogram() / Gauge() → Grafana dashboard",
    what: "Industry-standard infrastructure monitoring stack extended for ML. Prometheus collects time-series metrics; Grafana visualises them on dashboards. ML-specific metrics to track: prediction latency (p50/p99), request throughput, prediction distribution (histogram), model version currently serving, error rate.",
    when: "Production ML APIs in Kubernetes or Docker. When you need real-time operational monitoring (latency, throughput, error rates) alongside ML-specific metrics. When your organisation already uses Prometheus/Grafana for infrastructure monitoring.",
    pitfall: "Prometheus stores raw metric time series but not individual predictions — don't try to use it for data drift analysis. Use Evidently/NannyML for drift and Prometheus for operational metrics.",
    alternatives: "Datadog (SaaS, excellent ML integrations), New Relic, AWS CloudWatch with custom metrics, OpenTelemetry (vendor-neutral instrumentation standard).",
    code: `from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define ML-specific metrics:
PREDICTION_COUNTER   = Counter('model_predictions_total',
    'Total predictions made', ['model_version', 'prediction_class'])
PREDICTION_LATENCY   = Histogram('model_prediction_latency_seconds',
    'Prediction latency', buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0])
PREDICTION_PROBA     = Histogram('model_prediction_probability',
    'Distribution of predicted probabilities',
    buckets=[i/10 for i in range(11)])
MODEL_VERSION_GAUGE  = Gauge('model_version_loaded',
    'Currently loaded model version', ['version'])

# In FastAPI endpoint:
@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()
    prob = pipeline.predict_proba([request.features])[0, 1]
    latency = time.time() - start_time

    # Record metrics:
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_PROBA.observe(prob)
    PREDICTION_COUNTER.labels(
        model_version="v1.2",
        prediction_class=str(int(prob >= 0.42))
    ).inc()

    return {"probability": prob}

# Expose metrics endpoint for Prometheus scraping:
# GET /metrics  → Prometheus scrapes this every 15 seconds`,
    links: "FastAPI Model Server · Production Monitoring · Drift Detection",
  },

  // ═══════════════════════════════════════════
  // DRIFT DETECTION
  // ═══════════════════════════════════════════
  {
    name: "Population Stability Index (PSI)",
    group: "Drift Detection",
    stage: "📊 Monitor",
    lib: "manual (numpy) · nannyml",
    api: "psi = sum((actual_pct - expected_pct) * log(actual_pct / expected_pct))",
    what: "Financial industry standard for measuring distribution shift. Quantifies how much the distribution of a variable has changed between two periods. PSI < 0.1 = no significant change; 0.1-0.25 = moderate change (investigate); > 0.25 = major shift (model likely needs retraining).",
    when: "Credit scoring, insurance, fraud detection — any regulated industry where PSI is a standard reporting metric. As a complement to KS test for drift detection. For monitoring input feature distributions and predicted probability distributions over time.",
    pitfall: "PSI requires binning continuous variables — the choice of bin boundaries (usually defined on the reference/training distribution) significantly affects the PSI value. Always compute bins on reference data and apply the same bins to current data.",
    alternatives: "KS test (better statistical rigor), Wasserstein distance (more sensitive to location shifts), Jensen-Shannon divergence (symmetric, bounded).",
    code: `import numpy as np
import pandas as pd

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. PSI > 0.25 → significant drift."""
    # Define bins on reference distribution:
    bin_edges   = np.quantile(reference, np.linspace(0, 1, bins + 1))
    bin_edges[0] -= 0.001   # Include min value
    bin_edges[-1] += 0.001  # Include max value

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current,   bins=bin_edges)[0]

    # Convert to percentages (avoid division by zero):
    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current),   1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(psi, 4)

def monitor_features(train_df, prod_df, feature_cols):
    results = []
    for col in feature_cols:
        psi = compute_psi(train_df[col].dropna(), prod_df[col].dropna())
        status = "OK" if psi < 0.1 else "WARN" if psi < 0.25 else "ALERT"
        results.append({"feature": col, "psi": psi, "status": status})
    return pd.DataFrame(results).sort_values("psi", ascending=False)

drift_report = monitor_features(train_df, production_df, feature_cols)
print(drift_report[drift_report.status != "OK"])`,
    links: "Evidently AI · Covariate Shift (Problems) · Concept Drift (Problems)",
  },
  {
    name: "KS Test for Continuous Drift",
    group: "Drift Detection",
    stage: "📊 Monitor",
    lib: "scipy",
    api: "scipy.stats.ks_2samp(reference, current)",
    what: "Kolmogorov-Smirnov two-sample test measures the maximum absolute difference between two empirical CDFs. The p-value indicates whether the two samples come from the same distribution. More statistically rigorous than PSI but less interpretable to business stakeholders.",
    when: "Continuous numeric feature monitoring. Validating that production data matches training distribution before retraining. Comparing model probability distributions across time windows. Any automated pipeline that needs a formal statistical test for drift.",
    pitfall: "For large production datasets (> 100k samples), the KS test almost always rejects the null hypothesis even for trivially small distributional differences — the test becomes too sensitive. Use effect size (KS statistic, not just p-value) and PSI for large datasets.",
    alternatives: "PSI (more intuitive, financial industry standard), Chi-square test (for categorical features), Wasserstein distance (for more nuanced distribution comparison), MMD (for high-dimensional).",
    code: `from scipy import stats
import pandas as pd
import numpy as np

def run_drift_checks(reference_df, current_df, feature_cols, alpha=0.05):
    """Run KS drift tests. Returns features with statistically significant drift."""
    results = []
    for col in feature_cols:
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values

        ks_stat, p_value = stats.ks_2samp(ref, cur)
        drifted = (p_value < alpha) and (ks_stat > 0.1)  # Both statistical AND practical significance

        results.append({
            "feature":   col,
            "ks_stat":   round(ks_stat, 4),   # Effect size (0=identical, 1=completely different)
            "p_value":   round(p_value, 6),
            "drifted":   drifted,
        })

    return pd.DataFrame(results).sort_values("ks_stat", ascending=False)

drift_results = run_drift_checks(train_df, production_jan_df, feature_cols)
drifted_features = drift_results[drift_results.drifted]["feature"].tolist()

if len(drifted_features) > 0:
    print(f"⚠️  Drift detected in {len(drifted_features)} features: {drifted_features}")
    # Trigger retraining pipeline or alert
`,
    links: "Adversarial Validation (EDA table) · Covariate Shift (Problems) · Evidently AI",
  },

  // ═══════════════════════════════════════════
  // DEPLOYMENT STRATEGIES
  // ═══════════════════════════════════════════
  {
    name: "Blue-Green Deployment",
    group: "Deployment Strategies",
    stage: "🚀 Serve",
    lib: "Kubernetes / nginx / cloud load balancer",
    api: "Two identical production environments — switch traffic between them",
    what: "Maintains two identical production environments: Blue (current production) and Green (new version). Switch 100% of traffic from Blue to Green instantly once validation passes. If anything goes wrong, rollback by switching traffic back to Blue in seconds.",
    when: "Any major model update where you want zero-downtime deployment with instant rollback capability. When new model version has different preprocessing requirements that make a gradual rollout complex.",
    pitfall: "Blue-Green requires double the infrastructure while both environments are live. Feature store or database changes that aren't backward-compatible prevent a clean rollback — always ensure database changes are backward-compatible or blue-green won't work cleanly.",
    alternatives: "Rolling update (Kubernetes default, gradual replacement), Canary deployment (gradual traffic shift), Shadow deployment (copy traffic, don't use new predictions).",
    code: `# Kubernetes blue-green via service selector switching:

# Blue deployment (current production):
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-model-blue
spec:
  selector:
    matchLabels: {app: credit-model, version: blue}
  template:
    metadata:
      labels: {app: credit-model, version: blue}
    spec:
      containers:
      - image: credit-api:v1.1.0

---
# Green deployment (new version — deploy but no traffic):
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-model-green
spec:
  selector:
    matchLabels: {app: credit-model, version: green}
  template:
    metadata:
      labels: {app: credit-model, version: green}
    spec:
      containers:
      - image: credit-api:v1.2.0

---
# Service initially points to blue:
apiVersion: v1
kind: Service
metadata:
  name: credit-model-service
spec:
  selector: {app: credit-model, version: blue}  # ← Change to "green" to switch

# Switch traffic to green (after validation):
# kubectl patch service credit-model-service -p '{"spec":{"selector":{"version":"green"}}}'
# Rollback: change "green" back to "blue"`,
    links: "Canary Deployment · Kubernetes · CI/CD for ML · Model Registry",
  },
  {
    name: "Canary Deployment",
    group: "Deployment Strategies",
    stage: "🚀 Serve",
    lib: "Kubernetes + istio / nginx",
    api: "Traffic split: 95% → old model, 5% → new model",
    what: "Gradually shifts production traffic from the old model to the new one (e.g., 5% → 10% → 25% → 50% → 100%) while monitoring performance metrics at each stage. If the canary version performs worse, stop the rollout and revert the traffic split.",
    when: "Large-scale ML systems where even a small performance regression affects many users. When a model update is significant enough to warrant careful monitoring before full rollout. A/B testing different model architectures at the infrastructure level.",
    pitfall: "Canary requires comparing performance across two traffic cohorts — ensure the 5% canary receives a representative sample, not systematically different traffic (e.g., don't accidentally route only mobile users to the canary).",
    alternatives: "Blue-green (faster but no gradual validation), feature flags (application-level traffic splitting, no infrastructure changes needed), shadow deployment (zero risk, but requires double compute).",
    code: `# Nginx upstream weight-based canary:
# nginx.conf snippet:
upstream credit_model_backend {
    server blue-service:8000  weight=95;   # 95% → current production
    server green-service:8000 weight=5;    # 5%  → new model version
}

# Istio virtual service for Kubernetes traffic splitting:
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: credit-model
spec:
  http:
  - route:
    - destination:
        host: credit-model-blue
        port: {number: 8000}
      weight: 95
    - destination:
        host: credit-model-green
        port: {number: 8000}
      weight: 5

# Automated canary promotion script:
def advance_canary(current_weight, increment=10, max_weight=100):
    if evaluate_canary_metrics():  # Check AUC, error rate, latency vs production
        new_weight = min(current_weight + increment, max_weight)
        update_traffic_weight(new_weight)
        print(f"Canary promoted to {new_weight}%")
    else:
        rollback_canary()
        raise ValueError("Canary metrics degraded — rolled back")`,
    links: "Blue-Green Deployment · Production Monitoring · Kubernetes",
  },
  {
    name: "Shadow Deployment",
    group: "Deployment Strategies",
    stage: "🚀 Serve",
    lib: "nginx / envoy proxy / custom middleware",
    api: "Mirror 100% of production traffic to new model, discard predictions",
    what: "Runs the new model version in parallel with production, sending it a copy of all real traffic. The new model's predictions are logged but never returned to users. Allows real-world performance evaluation with zero user impact — the safest way to validate a new model.",
    when: "High-stakes predictions (medical diagnosis, financial decisions) where any prediction error has serious consequences. Testing a fundamentally different model architecture before exposing it to users. Any time you want real production data validation with no risk.",
    pitfall: "Shadow deployment doubles compute costs for the period it runs. Latency of the shadow model doesn't matter for users (predictions are discarded), so slow models appear to perform fine in shadow but would cause problems in production.",
    alternatives: "Canary (lower compute cost, gradual exposure), replay testing (replay historical traffic against new model offline — cheaper but misses real-time data distribution).",
    code: `# FastAPI middleware for shadow deployment:
from fastapi import FastAPI, Request
import asyncio, httpx, logging

app = FastAPI()
PRODUCTION_MODEL_URL = "http://blue-service:8000/predict"
SHADOW_MODEL_URL     = "http://green-service:8000/predict"

async def fire_shadow_request(payload: dict):
    """Send to shadow model and log — never block the main response."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            shadow_resp = await client.post(SHADOW_MODEL_URL, json=payload)
            shadow_pred = shadow_resp.json()
            # Log shadow prediction for later comparison:
            logging.info({
                "type": "shadow_prediction",
                "shadow_probability": shadow_pred["probability"],
            })
    except Exception as e:
        logging.warning(f"Shadow request failed: {e}")  # Don't propagate — never fail production

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()

    # Fire shadow request without awaiting (non-blocking):
    asyncio.create_task(fire_shadow_request(payload))

    # Return production model prediction immediately:
    async with httpx.AsyncClient() as client:
        prod_resp = await client.post(PRODUCTION_MODEL_URL, json=payload)
        return prod_resp.json()`,
    links: "Canary Deployment · Production Monitoring · CI/CD for ML",
  },
  {
    name: "Champion-Challenger Testing",
    group: "Deployment Strategies",
    stage: "🚀 Serve",
    lib: "application-level (any framework)",
    api: "Persistent user/entity assignment to champion or challenger cohort",
    what: "Routes a consistent percentage of user requests to a new model (Challenger) while the existing model (Champion) serves the majority. Unlike canary deployment, the same users/entities are consistently routed to the same model — ensuring user experience consistency and clean statistical comparison.",
    when: "Business-critical ML systems where you need statistically significant A/B test results before committing to a new model. When consistent user experience matters (showing the same model to a user across sessions). Credit scoring, pricing models, recommendation systems.",
    pitfall: "A/B significance requires adequate sample size — use a power analysis to determine how long to run the test and what traffic percentage to allocate. Running champion-challenger for too short a time leads to false conclusions from sampling noise.",
    alternatives: "Canary (simpler, no user consistency), multi-armed bandit (dynamic allocation based on observed performance — less interpretable but faster).",
    code: `import hashlib
from enum import Enum

class ModelVersion(Enum):
    CHAMPION   = "champion"
    CHALLENGER = "challenger"

def assign_model_version(entity_id: str, challenger_pct: float = 0.1) -> ModelVersion:
    """Consistently assign entities to model versions using deterministic hashing."""
    hash_val = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    bucket   = (hash_val % 100) / 100.0   # 0.0 to 1.0
    return ModelVersion.CHALLENGER if bucket < challenger_pct else ModelVersion.CHAMPION

# In prediction endpoint:
def predict_with_ab_test(customer_id: str, features: list):
    version = assign_model_version(customer_id, challenger_pct=0.1)
    model   = CHALLENGER_MODEL if version == ModelVersion.CHALLENGER else CHAMPION_MODEL
    prob    = model.predict_proba([features])[0, 1]

    # Log for A/B analysis:
    log_prediction(customer_id, prob, version.value)

    return {"probability": prob, "model_version": version.value}

# Statistical significance test (after collecting enough data):
from scipy import stats
champion_outcomes   = get_outcomes("champion")   # e.g., conversion rates
challenger_outcomes = get_outcomes("challenger")
stat, p_value = stats.mannwhitneyu(champion_outcomes, challenger_outcomes)
print(f"Champion vs Challenger p-value: {p_value:.4f}")`,
    links: "Canary Deployment · Production Monitoring · Statistical Tests (FE table)",
  },

  // ═══════════════════════════════════════════
  // PIPELINE ORCHESTRATION
  // ═══════════════════════════════════════════
  {
    name: "Prefect — Python-native Workflow Orchestration",
    group: "Pipeline Orchestration",
    stage: "🏗️ Orchestrate",
    lib: "prefect",
    api: "@flow / @task / flow.run() / prefect deploy",
    what: "Modern Python-native workflow orchestration. Define ML pipelines as Python functions decorated with @flow and @task. Provides retries, caching, observability, scheduling, and a UI for monitoring pipeline runs. Much simpler to adopt than Airflow for Python-centric teams.",
    when: "Automating the full ML retraining pipeline: data ingestion → feature engineering → training → evaluation → deployment. Scheduled batch scoring pipelines. Any repeated ML workflow that needs retry logic, observability, and scheduling.",
    pitfall: "Prefect 2.x/3.x is a complete rewrite from 1.x — don't mix documentation versions. Prefect tasks that take > 10s benefit from persist_result=True caching so failed pipelines resume from the last successful task rather than restarting from scratch.",
    alternatives: "Apache Airflow (more mature, larger ecosystem, steeper learning curve), Dagster (data-asset-centric, excellent for dbt integration), Metaflow (ML-specific, Netflix-developed), ZenML.",
    code: `from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(retries=3, retry_delay_seconds=60, cache_key_fn=task_input_hash,
      cache_expiration=timedelta(hours=24))
def load_training_data(version: str):
    """Load and validate training data. Cached for 24h so retraining doesn't re-download."""
    df = pd.read_parquet(f"s3://data-lake/features/v{version}/train.parquet")
    validate_schema(df)
    return df

@task(retries=2)
def train_model(df, params: dict):
    """Train model and return OOF score."""
    oof_preds, oof_score = run_oof(lgbm_model(**params), df, cv=cv)
    mlflow.log_metrics({"oof_auc": oof_score})
    return oof_score, oof_preds

@task
def evaluate_and_gate(oof_score: float, threshold: float = 0.85):
    """Fail the pipeline if new model doesn't meet performance gate."""
    if oof_score < threshold:
        raise ValueError(f"Model failed quality gate: AUC {oof_score:.4f} < {threshold}")
    return True

@task
def deploy_model(model_uri: str):
    """Register model and promote to staging."""
    client = MlflowClient()
    client.transition_model_version_stage(name="credit_model", version=..., stage="Staging")

@flow(name="weekly-model-retraining", log_prints=True)
def retraining_pipeline(data_version: str = "latest", deploy: bool = True):
    df         = load_training_data(data_version)
    score, _   = train_model(df, params=BEST_PARAMS)
    passed     = evaluate_and_gate(score)
    if passed and deploy:
        deploy_model(model_uri="...")
    return score

# Schedule: retraining_pipeline.serve(cron="0 2 * * 1")  # Every Monday 2am`,
    links: "CI/CD for ML · MLflow Tracking · Drift Detection · Retraining & Governance",
  },
  {
    name: "Apache Airflow",
    group: "Pipeline Orchestration",
    stage: "🏗️ Orchestrate",
    lib: "apache-airflow",
    api: "DAG() / @task / BashOperator / PythonOperator",
    what: "Industry-standard batch workflow orchestrator. Defines pipelines as Directed Acyclic Graphs (DAGs) with tasks as nodes. Mature ecosystem with 1000+ pre-built operators for every data warehouse, cloud provider, and ML platform. The standard at large enterprises.",
    when: "Enterprise environments with complex multi-system pipelines (data warehouse → feature engineering → training → deployment → monitoring). When your organisation already uses Airflow for data engineering and wants to add ML pipelines. High-availability requirements.",
    pitfall: "Airflow's scheduler uses a relational database for state management — scaling it requires a proper Airflow deployment (Kubernetes, Celery executor). Don't run heavy ML training directly inside Airflow tasks — use KubernetesPodOperator or SageMaker operator to offload training to separate compute.",
    alternatives: "Prefect (simpler for Python-centric teams), Dagster (better asset lineage), Kubeflow Pipelines (Kubernetes-native, ML-specific), Metaflow.",
    code: `from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="ml_retraining_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * 1",          # Every Monday 2am
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
        "email": ["ml-team@company.com"],
    },
) as dag:

    @task
    def validate_data():
        result = run_great_expectations_suite("training_data_suite")
        if not result["success"]:
            raise ValueError("Data validation failed")
        return result

    @task
    def train_model(validation_result: dict):
        run_id = trigger_training_job()  # Offload to separate compute
        return run_id

    @task
    def evaluate_model(run_id: str):
        metrics = get_run_metrics(run_id)
        if metrics["oof_auc"] < 0.85:
            raise ValueError(f"Model quality gate failed: {metrics}")
        return metrics

    @task
    def deploy_to_staging(metrics: dict):
        register_and_promote_model(stage="Staging")

    # DAG wiring:
    validation  = validate_data()
    run_id      = train_model(validation)
    metrics     = evaluate_model(run_id)
    deploy_to_staging(metrics)`,
    links: "Prefect · CI/CD for ML · MLflow Tracking · Data Validation",
  },

  // ═══════════════════════════════════════════
  // RETRAINING & GOVERNANCE
  // ═══════════════════════════════════════════
  {
    name: "Automated Retraining Triggers",
    group: "Retraining & Governance",
    stage: "🔄 Automate",
    lib: "Evidently · NannyML · custom (cron + logic)",
    api: "if psi > 0.25 or estimated_auc < threshold: trigger_retraining()",
    what: "Systematic criteria that automatically initiate model retraining when specific conditions are met — rather than fixed schedules that may retrain unnecessarily or miss urgent degradation. Three trigger types: time-based (simplest), performance-based (retrain when metrics degrade), and drift-based (retrain when input distribution shifts).",
    when: "Any production model in a changing environment. Fraud detection (new fraud patterns emerge), demand forecasting (seasonal shifts), recommendation systems (user behavior changes). Start with time-based triggers, graduate to drift-based as monitoring matures.",
    pitfall: "Automated retraining without a quality gate (automated evaluation against hold-out metrics before deployment) is dangerous — a bad training data batch could auto-deploy a worse model. Always include a gated deployment step.",
    alternatives: "Manual retraining on a fixed schedule (simpler to reason about, but wasteful or insufficient depending on drift rate). Continuous learning / online learning (more complex but eliminates discrete retraining cycles).",
    code: `# Automated retraining decision logic:
import pandas as pd
from datetime import datetime

class RetrainingDecider:
    def __init__(self, psi_threshold=0.25, perf_threshold=0.05,
                 max_days_without_retrain=30):
        self.psi_threshold   = psi_threshold
        self.perf_threshold  = perf_threshold
        self.max_days        = max_days_without_retrain

    def should_retrain(self, monitoring_report: dict) -> tuple[bool, str]:
        reasons = []

        # Trigger 1: Data drift above threshold
        max_psi = max(f["psi"] for f in monitoring_report["features"])
        if max_psi > self.psi_threshold:
            reasons.append(f"PSI={max_psi:.3f} exceeds threshold {self.psi_threshold}")

        # Trigger 2: Estimated performance degradation
        est_auc = monitoring_report.get("estimated_auc")
        prod_auc = monitoring_report.get("production_auc_at_deployment")
        if est_auc and prod_auc and (prod_auc - est_auc) > self.perf_threshold:
            reasons.append(f"Performance degraded: {prod_auc:.3f} → {est_auc:.3f}")

        # Trigger 3: Time-based fallback (force retrain every N days)
        days_since = (datetime.now() - monitoring_report["last_retrain_date"]).days
        if days_since >= self.max_days:
            reasons.append(f"Scheduled: {days_since} days since last retrain")

        should = len(reasons) > 0
        return should, "; ".join(reasons) if reasons else "No retraining needed"

# In monitoring pipeline:
decider  = RetrainingDecider()
retrain, reason = decider.should_retrain(latest_monitoring_report)
if retrain:
    trigger_retraining_pipeline(reason=reason)
    notify_team(f"Retraining triggered: {reason}")`,
    links: "Evidently AI · NannyML · Prefect · Drift Detection",
  },
  {
    name: "Model Cards & Documentation",
    group: "Retraining & Governance",
    stage: "📦 Package",
    lib: "model-card-toolkit · huggingface hub · manual",
    api: "ModelCard(model_details=...).export_format()",
    what: "Structured documentation that accompanies every deployed model: intended use, training data description, evaluation results across demographic groups, limitations, out-of-scope uses, and bias analysis. Required for responsible AI deployment and increasingly mandated by AI regulations (EU AI Act).",
    when: "Any model deployed in a context affecting people (lending, healthcare, hiring, criminal justice). Enterprise deployments where governance and auditability are required. Any public-facing model. Best practice for all production models regardless of regulatory requirement.",
    pitfall: "Model cards that describe a model as of its initial deployment become outdated after retraining. Version your model cards alongside your model versions — each Model Registry entry should reference its model card version.",
    alternatives: "Datasheets for Datasets (for data documentation), SHAP/LIME for interpretability documentation, custom templated Word/Markdown documentation.",
    code: `# Example model card structure (in YAML/JSON for machine-readability):
model_card = {
    "model_details": {
        "name": "Credit Default Prediction Model",
        "version": "1.2.0",
        "type": "Binary Classifier (LightGBM)",
        "train_date": "2025-01-15",
        "contact": "ml-team@company.com",
    },
    "intended_use": {
        "primary_uses": ["Credit application screening", "Risk scoring"],
        "primary_users": ["Underwriting team", "Automated credit workflow"],
        "out_of_scope": [
            "Decisions about individuals in GDPR-regulated regions without human review",
            "Scoring applicants under 18 years old",
        ],
    },
    "training_data": {
        "source": "Internal loan applications 2020-2024",
        "n_samples": 1_500_000,
        "features": 42,
        "positive_rate": "8.3% (defaults)",
        "demographic_note": "Not trained on race/ethnicity/religion features",
    },
    "evaluation": {
        "overall": {"roc_auc": 0.872, "f1": 0.61, "precision": 0.67, "recall": 0.56},
        "by_age_group": {
            "18-30": {"roc_auc": 0.841},
            "30-50": {"roc_auc": 0.884},
            "50+":   {"roc_auc": 0.869},
        },
    },
    "limitations": [
        "Performance degrades on applicants with < 2 years of credit history",
        "Requires retraining if interest rate environment changes significantly",
    ],
    "ethical_considerations": "Model is one input to underwriting decisions; human review required for all borderline cases (score 40-60).",
}`,
    links: "Model Registry · MLflow Tracking · Production Monitoring",
  },

  // ═══════════════════════════════════════════
  // MLOPS ANTI-PATTERNS
  // ═══════════════════════════════════════════
  {
    name: "Training-Serving Skew",
    group: "MLOps Anti-Patterns",
    stage: "🔴 Anti-Pattern",
    lib: "Feature Stores · strict pipeline versioning",
    api: "Prevention: use Feature Store or version-controlled transform code",
    what: "The most common cause of silent production failures. Occurs when the features computed during serving differ from those during training — different code paths, different preprocessing logic, different missing value handling, or different feature computation windows.",
    detect: "Compare feature statistics logged at training time with features arriving at the inference endpoint using Evidently or manual logging. If distributions differ despite similar raw data, skew is present.",
    fix: "Use a Feature Store (Feast, Tecton) so features are defined once and served identically to training and inference. If no feature store, version your feature transformation code separately from model code and deploy them together. Test by running the same raw data through both training and inference pipelines and comparing features at every step.",
    severity: "Critical — a skewed model silently makes wrong predictions for its entire deployment lifetime without any obvious errors.",
    code: `# Detecting training-serving skew:
# Log feature statistics at training time:
training_feature_stats = {
    col: {"mean": df[col].mean(), "std": df[col].std(),
          "null_pct": df[col].isnull().mean(), "min": df[col].min(), "max": df[col].max()}
    for col in feature_cols
}
# Save with the model:
# joblib.dump({"model": model, "feature_stats": training_feature_stats}, "model_v1.joblib")

# In serving — log incoming feature stats periodically:
def log_serving_stats(X_batch: np.ndarray, feature_names: list):
    for i, col in enumerate(feature_names):
        serving_mean = X_batch[:, i].mean()
        training_mean = training_feature_stats[col]["mean"]
        rel_drift = abs(serving_mean - training_mean) / (abs(training_mean) + 1e-9)
        if rel_drift > 0.3:  # 30% relative change in mean
            log_alert(f"Potential skew in {col}: train_mean={training_mean:.3f}, serving_mean={serving_mean:.3f}")`,
    links: "Feature Stores · Data Validation · Drift Detection · Evidently AI",
  },
  {
    name: "No Model Versioning or Rollback Plan",
    group: "MLOps Anti-Patterns",
    stage: "🔴 Anti-Pattern",
    lib: "MLflow Registry · Model Card · Git",
    api: "Prevention: always register every deployed model with a version",
    what: "Deploying a new model by overwriting the previous one — no version history, no ability to roll back when the new model performs worse. Incredibly common in early-stage ML systems. The ML equivalent of deploying without git.",
    detect: "If you can't answer 'what model is currently in production and which exact training run produced it' in 30 seconds, you have this problem.",
    fix: "Register every model in a Model Registry (MLflow, W&B, custom S3 metadata) before deployment. Use semantic versioning (1.2.0). Keep the previous production version deployable for at least 30 days. Test rollback in staging before you need it in production.",
    severity: "High — without versioning, every deployment is a one-way door. Discovering performance degradation with no rollback option is an incident.",
    code: `# Minimum viable model versioning — no fancy infrastructure required:
import boto3, json
from datetime import datetime

def deploy_model_with_versioning(model_path, metrics, previous_version_path):
    """Register and deploy model while preserving rollback capability."""
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    # Save current production model as 'previous' BEFORE deploying new one:
    s3 = boto3.client('s3')
    s3.copy_object(
        CopySource={'Bucket': 'ml-models', 'Key': 'production/model.joblib'},
        Bucket='ml-models',
        Key=f'archive/{version}_previous.joblib',  # Never delete this
    )

    # Deploy new model:
    s3.upload_file(model_path, 'ml-models', 'production/model.joblib')

    # Write version metadata:
    metadata = {"version": version, "metrics": metrics, "deployed_at": str(datetime.now())}
    s3.put_object(Bucket='ml-models', Key='production/metadata.json',
                  Body=json.dumps(metadata))

    # Rollback (when needed):
    # s3.copy_object(CopySource={'Bucket':'ml-models','Key':f'archive/{version}_previous.joblib'},
    #                Bucket='ml-models', Key='production/model.joblib')`,
    links: "Model Registry · Blue-Green Deployment · MLflow Tracking",
  },
  {
    name: "No Performance Monitoring After Deployment",
    group: "MLOps Anti-Patterns",
    stage: "🔴 Anti-Pattern",
    lib: "Evidently · NannyML · Prometheus",
    api: "Prevention: monitor from day 1, not after the first incident",
    what: "Deploying a model and assuming it will continue performing well indefinitely. Without monitoring, model degradation is discovered by users or business impact — hours, days, or months after it began. This is the most common MLOps failure mode in production.",
    detect: "If you haven't looked at your model's production performance metrics in the past week, you may have this problem. Business metrics (conversion rate, customer complaints) decreasing unexpectedly is often the first symptom.",
    fix: "Implement at least three monitoring layers: (1) operational metrics (latency, error rate) via Prometheus, (2) data drift monitoring (PSI/KS on input features) via Evidently, (3) performance estimation without labels via NannyML. Set automated alerts on thresholds. Monitor from the first day of deployment.",
    severity: "Critical — silent model degradation is one of the most costly ML failures because it can persist undetected for extended periods.",
    code: `# Minimum viable production monitoring — can be implemented in one day:
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Log every prediction with input features:
def predict_with_logging(features: dict) -> dict:
    X = preprocess(features)
    prob = model.predict_proba(X)[0, 1]

    # Log to database/S3/BigQuery for later analysis:
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "probability": float(prob),
        "prediction":  int(prob >= THRESHOLD),
        **{f"feat_{k}": v for k, v in features.items()},  # Log all features
    }
    logging.info(json.dumps(log_entry))   # Goes to your log aggregator

    return {"probability": prob, "prediction": int(prob >= THRESHOLD)}

# Weekly monitoring script (run as cron job or Prefect task):
def weekly_health_check():
    recent_preds = load_last_week_predictions()
    reference    = load_training_distribution()

    drift_report = monitor_features(reference, recent_preds, feature_cols)
    pred_dist    = recent_preds["probability"].describe()

    if (drift_report.status == "ALERT").any():
        send_slack_alert(f"Feature drift detected: {drift_report[drift_report.status=='ALERT'].to_string()}")

    if pred_dist["mean"] < 0.02 or pred_dist["mean"] > 0.5:   # Suspiciously low or high positive rate
        send_slack_alert(f"Prediction distribution anomaly: mean proba = {pred_dist['mean']:.3f}")`,
    links: "Evidently AI · NannyML · Prometheus + Grafana · Drift Detection",
  },
];

// ─── COMPONENT ───────────────────────────────────────────────────────────────
const ALL_GROUPS = ["All", ...Object.keys(GROUP_ACCENT)];
const ALL_STAGES = ["All", ...Object.keys(STAGE_STYLE)];

function StageBadge({ s }) {
  const st = STAGE_STYLE[s] || { bg:"var(--border-subtle)", color:"#888", border:"#888" };
  return <span style={{ display:"inline-block", padding:"2px 8px", borderRadius:"4px", fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.04em", background:st.bg, color:st.color, border:`1px solid ${st.border}44`, whiteSpace:"nowrap" }}>{s}</span>;
}
function LibBadge({ text }) {
  return <span style={{ display:"inline-block", padding:"1px 6px", borderRadius:"3px", fontSize:"0.61rem", fontWeight:600, background:"#0d0d1c", color:"#5a5a88", border:"1px solid var(--border-default)", whiteSpace:"nowrap", margin:"1px" }}>{text}</span>;
}

function Card({ icon, label, text, accent }) {
  return (
    <div style={{ padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:`1px solid ${accent}20` }}>

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

const tdS = { padding:"10px 12px", verticalAlign:"middle", borderBottom:"1px solid var(--bg-surface)", color:"var(--text-primary)", fontSize:"0.81rem" };
const thS = { padding:"10px 12px", textAlign:"left", fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:"var(--text-tertiary)", borderBottom:"2px solid var(--border-faint)", background:"var(--bg-base)", position:"sticky", zIndex:5 };

function Row({ item, idx }) {
  const [open, setOpen] = useState(false);
  const accent = GROUP_ACCENT[item.group] || "#888";

  const isAntiPattern = item.group === "MLOps Anti-Patterns";

  return (
    <>
      <tr onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background="var(--bg-surface)"}
        onMouseLeave={e => e.currentTarget.style.background=idx%2===0?"var(--bg-surface)":"var(--bg-surface)"}
        style={{ cursor:"pointer", background:idx%2===0?"var(--bg-surface)":"var(--bg-surface)", borderLeft:`3px solid ${accent}`, transition:"background 0.12s" }}>
        <td style={tdS}>{open?"▾":"▸"}</td>
        <td style={{ ...tdS, fontWeight:700, color:accent, fontSize:"0.87rem" }}>{item.name}</td>
        <td style={tdS}>{item.lib.split("·").slice(0,2).map(l=><LibBadge key={l} text={l.trim()}/>)}</td>
        <td style={tdS}><StageBadge s={item.stage}/></td>
        <td style={{ ...tdS, fontFamily:"var(--font-mono)", fontSize:"0.66rem", color:"var(--text-dim)", maxWidth:"220px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{item.api}</td>
      </tr>
      {open && (
        <tr style={{ background:"var(--bg-surface)", borderLeft:`3px solid ${accent}` }}>
          <td colSpan={5} style={{ padding:"0 0 0 16px" }}>
            <div style={{ padding:"14px 16px 14px 0" }}>
              {isAntiPattern ? (
                <>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                    <Card icon="🧨" label="What It Is" text={item.what} accent={accent}/>
                    <Card icon="🔍" label="How to Detect" text={item.detect} accent="#fb923c"/>
                  </div>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                    <Card icon="🔧" label="Fix" text={item.fix} accent="#4ade80"/>
                    <Card icon="⚠️" label="Severity" text={item.severity} accent="#f87171"/>
                  </div>
                </>
              ) : (
                <>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                    <Card icon="📋" label="What It Does" text={item.what} accent={accent}/>
                    <Card icon="✅" label="Use When" text={item.when} accent="#4ade80"/>
                  </div>
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                    <Card icon="⚠️" label="Common Pitfall" text={item.pitfall} accent="#fb923c"/>
                    <Card icon="🔄" label="Alternatives" text={item.alternatives} accent="#818cf8"/>
                  </div>
                </>
              )}
              <CodeBlock code={item.code}/>
              {item.links && (
                <div style={{ marginTop:"10px", padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:"1px solid #818cf820" }}>
                  <div style={{ fontSize:"0.62rem", fontWeight:700, letterSpacing:"0.09em", color:"#818cf8", textTransform:"uppercase", marginBottom:"6px" }}>🔗 See Also</div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:"4px" }}>
                    {item.links.split("·").map(t => (
                      <span key={t} style={{ padding:"2px 8px", borderRadius:"12px", fontSize:"0.63rem", fontWeight:600, background:"#818cf818", color:"#818cf8", border:"1px solid #818cf830" }}>{t.trim()}</span>
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

export default function Deploy_app() {
  const [search, setSearch] = useState("");
  const [grp, setGrp]       = useState("All");
  const [stg, setStg]       = useState("All");

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return ITEMS.filter(it => {
      const mg = grp === "All" || it.group === grp;
      const ms = stg === "All" || it.stage === stg;
      const mq = !q || [it.name, it.group, it.lib, it.what, it.when||"", it.pitfall||"", it.links||"", it.detect||"", it.fix||""].some(s => s.toLowerCase().includes(q));
      return mg && ms && mq;
    });
  }, [search, grp, stg]);

  const groups = useMemo(() => {
    const g = {};
    for (const it of filtered) { if (!g[it.group]) g[it.group]=[]; g[it.group].push(it); }
    return g;
  }, [filtered]);

  const stageCounts = useMemo(() => {
    const c = {};
    for (const it of ITEMS) c[it.stage] = (c[it.stage]||0)+1;
    return c;
  }, []);

  return (
    <div>
    <Header />

        <div style={{ fontFamily:"var(--font-body)", background:"var(--bg-base)", minHeight:"100vh", color:"var(--text-primary)" }}>
      
      {/* HEADER */}
      <div style={{ position:"sticky", top:"var(--header-h)", zIndex:10, background:"rgba(14,13,12,0.93)", backdropFilter:"blur(10px)", borderBottom:"1px solid var(--border-faint)", padding:"10px 20px 8px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:"14px", marginBottom:"8px", flexWrap:"wrap" }}>
          <div>
            <div style={{ fontSize:"1.05rem", fontWeight:700, color:"#fff", letterSpacing:"-0.02em" }}>
              <span style={{ color:"#34d399" }}>MLOps</span>
              <span style={{ color:"var(--text-dim)", margin:"0 6px" }}>·</span>
              <span style={{ color:"#60a5fa" }}>Deployment</span>
              <span style={{ color:"var(--text-dim)", margin:"0 6px" }}>·</span>
              <span style={{ color:"#f87171" }}>Monitoring</span>
              <span style={{ color:"var(--text-dim)", margin:"0 6px" }}>·</span>
              <span style={{ color:"#facc15" }}>Governance</span>
            </div>
            <div style={{ fontSize:"0.62rem", color:"var(--text-dim)", letterSpacing:"0.05em", marginTop:"1px" }}>
              {ITEMS.length} entries · 15 categories · production ML lifecycle · click any row to expand
            </div>
          </div>
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search: drift, serving, docker, canary, mlflow…"
            style={{ flex:1, minWidth:"180px", maxWidth:"320px", marginLeft:"auto", background:"#0c0c1c", border:"1px solid var(--border-default)", borderRadius:"6px", padding:"7px 11px", color:"var(--text-primary)", fontSize:"0.79rem", outline:"none" }}/>
          <div style={{ fontSize:"0.68rem", color:"var(--text-dim)", whiteSpace:"nowrap" }}>{filtered.length} shown</div>
        </div>

        {/* Stage filter */}
        <div style={{ display:"flex", gap:"4px", flexWrap:"wrap", marginBottom:"4px" }}>
          <span style={{ fontSize:"0.6rem", color:"var(--text-dim)", alignSelf:"center", marginRight:"4px" }}>STAGE:</span>
          {ALL_STAGES.map(s => {
            const active = stg === s; const st = STAGE_STYLE[s]||{color:"#666",border:"#666",bg:"#111"};
            return <button key={s} onClick={()=>setStg(s)} style={{ padding:"2px 9px", borderRadius:"4px", fontSize:"0.65rem", fontWeight:active?700:400, border:active?`1px solid ${st.border}`:"1px solid var(--border-subtle)", background:active?st.bg:"transparent", color:active?st.color:"var(--text-tertiary)" }}>
              {s}{s!=="All"&&stageCounts[s]?` (${stageCounts[s]})`:""}</button>;
          })}
        </div>

        {/* Group filter */}
        <div style={{ display:"flex", gap:"4px", flexWrap:"wrap" }}>
          <span style={{ fontSize:"0.6rem", color:"var(--text-dim)", alignSelf:"center", marginRight:"4px" }}>CATEGORY:</span>
          {ALL_GROUPS.map(g => {
            const active = grp === g; const accent = GROUP_ACCENT[g]||"#666";
            return <button key={g} onClick={()=>setGrp(g)} style={{ padding:"2px 9px", borderRadius:"4px", fontSize:"0.64rem", fontWeight:active?700:400, border:active?`1px solid ${accent}`:"1px solid var(--border-subtle)", background:active?`${accent}22`:"transparent", color:active?accent:"var(--text-tertiary)" }}>{g}</button>;
          })}
        </div>
      </div>

      {/* LIFECYCLE BANNER */}
      <div style={{ margin:"10px 20px 0", padding:"10px 16px", borderRadius:"6px", background:"var(--bg-surface)", border:"1px solid #1a1a34", fontSize:"0.72rem", color:"#50508a", display:"flex", gap:"6px", alignItems:"center", flexWrap:"wrap" }}>
        {[["📦","Serialize","Model Serialization + Registry"],["🔬","Track","Experiment Tracking"],["🚀","Serve","APIs + BentoML + Triton"],["🐳","Infra","Docker + Kubernetes"],["🔄","Automate","CI/CD + DVC + Orchestration"],["📊","Monitor","Evidently + NannyML + Prometheus"],["🔴","Avoid","MLOps Anti-Patterns"]].map(([icon, label, tooltip], i) => (
          <span key={label} style={{ display:"flex", alignItems:"center", gap:"5px", whiteSpace:"nowrap" }}>
            {i>0 && <span style={{ color:"var(--bg-overlay)", margin:"0 4px" }}>→</span>}
            <span title={tooltip} style={{ cursor:"help" }}>{icon} <span style={{ color:"var(--text-secondary)" }}>{label}</span></span>
          </span>
        ))}
      </div>

      {/* TABLE */}
      <div style={{ overflowX:"auto", marginTop:"10px" }}>
        <table style={{ width:"100%", borderCollapse:"collapse", minWidth:"820px" }}>
          <thead>
            <tr>
              <th style={{...thS, width:"24px"}}></th>
              <th style={thS}>Name</th>
              <th style={thS}>Library</th>
              <th style={thS}>Stage</th>
              <th style={{...thS, maxWidth:"220px"}}>API / Command</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([group, items]) => (
              <>
                <tr key={group+"_h"}>
                  <td colSpan={5} style={{ padding:"8px 14px 4px", fontSize:"0.65rem", fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", color:GROUP_ACCENT[group]||"#5555a0", background:"var(--bg-surface)", borderTop:"2px solid var(--bg-surface)", borderBottom:"1px solid var(--bg-surface)" }}>
                    ▪ {group} <span style={{ fontWeight:400, color:"#181838", marginLeft:6 }}>({items.length})</span>
                  </td>
                </tr>
                {items.map((item, i) => <Row key={item.name} item={item} idx={i}/>)}
              </>
            ))}
            {filtered.length===0 && <tr><td colSpan={5} style={{ padding:"60px", textAlign:"center", color:"var(--text-dim)" }}>No entries match. Try a different search or reset filters.</td></tr>}
          </tbody>
        </table>
      </div>
      <div style={{ padding:"16px", textAlign:"center", fontSize:"0.62rem", color:"var(--border-faint)", borderTop:"1px solid var(--bg-surface)" }}>
        MLflow · Weights & Biases · DVC · FastAPI · BentoML · Triton · Docker · Kubernetes · Great Expectations · Pandera · Evidently · NannyML · Prometheus · Grafana · Feast · Prefect · Airflow
      </div>
    </div></div>
  );
}