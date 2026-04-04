# Prompt Registry

Prompts are versioned like code. Each prompt version is stored as YAML with:

| Field | Purpose |
|:---|:---|
| `version` | Immutable prompt version (for example, `v1`, `v2`) |
| `created_date` | Auditability for release timeline |
| `system_prompt` | Base behavior and constraints |
| `user_template` | Runtime input template |
| `eval_scores` | Quality metrics tied to that exact prompt version |

Why this pattern works in production:

| Need | Registry behavior |
|:---|:---|
| Reproducibility | Every run can pin `name + version` |
| Safe iteration | New versions can be tested without overwriting old prompts |
| Rollback | If quality drops, load previous version immediately |
| Traceability | Eval scores live next to prompt text and metadata |

## Usage

```python
from prompt_registry.registry import PromptRegistry

registry = PromptRegistry()

prompt = registry.load("rag_synthesizer", "v1")
print(prompt["system_prompt"])

print(registry.list_versions("query_rewriter"))
```

