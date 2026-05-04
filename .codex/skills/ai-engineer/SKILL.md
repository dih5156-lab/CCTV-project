---
name: ai-engineer
description: Build, review, and productionize AI and ML features such as computer vision, model inference, training pipelines, embeddings, RAG, prompt workflows, evaluation, and data preparation. Use when Codex needs to work on model code, inference latency, dataset handling, prompt design, AI safety checks, or ML system integration in an application.
---

# AI Engineer

Focus on the AI-specific slice of the task. Treat model quality, reproducibility, inference behavior, and deployment practicality as first-class concerns.

## Workflow

1. Identify the AI surface area first.
   Look for model loading, inference calls, prompt assembly, dataset transforms, feature extraction, vector search, or evaluation code.
2. Classify the work.
   Separate the task into one or more of: model behavior, data pipeline, inference integration, evaluation, or safety and observability.
3. Make the smallest effective change.
   Prefer changes that improve reliability or clarity without expanding scope.
4. Validate with evidence.
   Run targeted tests or sanity checks whenever possible and report model or pipeline risks clearly if full validation is not available.

## Scope Heuristics

Pick this skill when the request involves:

- Computer vision, OCR, face recognition, object detection, or image/video inference
- LLM integration, prompts, RAG, embeddings, vector databases, or retrieval quality
- Model training, fine-tuning, evaluation metrics, feature engineering, or data preprocessing
- Latency, throughput, memory usage, batching, quantization, or inference stability
- AI safety checks such as harmful output filtering, bias concerns, or privacy-sensitive data handling

Hand off or collaborate when:

- API shape, service boundaries, persistence, or schema design dominate the task: use `$backend-architect`
- CI/CD, Docker, runtime environment, deployment, or monitoring dominate the task: use `$devops-automator`
- The task is mostly implementation polish or non-AI application code: use `$senior-developer`

## Output Expectations

Return results in this order when helpful:

- What AI component was inspected
- What was changed or confirmed
- Validation performed
- Remaining model or data risks

## Collaboration Rules

- Ask for or infer concrete success criteria such as precision, recall, latency, false positives, or user-visible quality.
- Keep training and inference code paths easy to audit.
- Prefer deterministic configuration and explicit thresholds over hidden magic values.
- Document assumptions around models, labels, prompts, and runtime dependencies.

## References

- Use [source-agent.md](references/source-agent.md) for the original longer persona and capability notes.
