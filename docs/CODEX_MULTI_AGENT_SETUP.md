# Codex Multi-Agent Setup

This project now has four reusable skill definitions staged under `.codex-skills/`.

## Skill Mapping

- `$ai-engineer`: model code, inference, CV, OCR, RAG, embeddings, prompt and dataset work
- `$backend-architect`: API design, schema, caching, concurrency, service boundaries, reliability
- `$devops-automator`: Docker, CI/CD, deployment, monitoring, environment and infra automation
- `$senior-developer`: feature implementation, bug fixes, refactors, polish, test updates

## Recommended Prompt Pattern

Use prompts like:

```text
이 작업을 멀티에이전트로 진행해줘.
- $backend-architect: 구조와 API 영향 검토
- $senior-developer: 실제 코드 수정
- $devops-automator: 배포/운영 영향 점검
최종 결과는 하나로 통합해줘.
```

## Routing Rule

- AI/ML가 핵심이면 `$ai-engineer`
- 구조/성능/스키마가 핵심이면 `$backend-architect`
- 배포/자동화/모니터링이 핵심이면 `$devops-automator`
- 구현과 마무리가 핵심이면 `$senior-developer`

## Notes

- `description` is the main trigger surface for a skill.
- `agents/openai.yaml` controls the UI-facing name and default prompt.
- The skill folder should live under `C:\Users\dih51\.codex\skills` for automatic discovery.
