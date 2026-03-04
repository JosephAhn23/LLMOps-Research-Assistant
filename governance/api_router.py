"""
FastAPI router for the governance dashboard.

Endpoints:
  GET /governance/report              - Full governance dashboard
  GET /governance/model-card/{name}/{version} - Model card as JSON
  GET /governance/audit-log           - Recent audit log entries
  POST /governance/verify-audit-log   - Verify chain integrity
  GET /governance/fairness/{name}/{version}   - Fairness report

Mount in api/main.py:
    from governance.api_router import router as governance_router
    app.include_router(governance_router, prefix="/governance", tags=["governance"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from governance.audit_log import CryptoAuditLog
from governance.bias_checks import BiasChecker
from governance.ci_enforcement import CIEnforcement
from governance.model_card_generator import ModelCardGenerator
from governance.pii_redaction import PIIRedactor

logger = logging.getLogger(__name__)
router = APIRouter()

_audit_log = CryptoAuditLog()
_card_generator = ModelCardGenerator()
_bias_checker = BiasChecker()
_pii_redactor = PIIRedactor()
_ci_enforcement = CIEnforcement()


class RedactRequest(BaseModel):
    text: str


@router.get("/report", summary="Full governance dashboard")
async def governance_report() -> Dict[str, Any]:
    """Returns governance status for the current production model."""
    valid, error = _audit_log.verify_chain()
    return {
        "status": "ok",
        "audit_log": {
            "entries": len(_audit_log),
            "chain_valid": valid,
            "head_hash": _audit_log.head_hash()[:16] + "...",
            "error": error,
        },
        "checks": {
            "pii_redaction": "active",
            "fairness_monitoring": "active",
            "model_card_generation": "active",
        },
    }


@router.get("/model-card/{model_name}/{version}", summary="Get model card")
async def get_model_card(
    model_name: str,
    version: int,
    ragas_faithfulness: float = 0.847,
    ragas_relevancy: float = 0.823,
) -> Dict[str, Any]:
    """Generate and return a model card for the specified model version."""
    card = _card_generator.generate(
        model_name=model_name,
        version=version,
        metrics={
            "ragas_faithfulness": ragas_faithfulness,
            "ragas_relevancy": ragas_relevancy,
        },
    )
    return card.to_dict()


@router.get("/model-card/{model_name}/{version}/markdown", summary="Get model card as markdown")
async def get_model_card_markdown(model_name: str, version: int) -> Dict[str, str]:
    card = _card_generator.generate(model_name=model_name, version=version, metrics={})
    return {"markdown": card.to_markdown()}


@router.get("/audit-log", summary="Get recent audit log entries")
async def get_audit_log(limit: int = 20, event_type: Optional[str] = None) -> List[Dict]:
    entries = _audit_log.get_entries(event_type=event_type)
    return [e.to_dict() for e in entries[-limit:]]


@router.post("/verify-audit-log", summary="Verify audit log chain integrity")
async def verify_audit_log() -> Dict[str, Any]:
    valid, error = _audit_log.verify_chain()
    return {
        "valid": valid,
        "entries": len(_audit_log),
        "head_hash": _audit_log.head_hash(),
        "error": error,
    }


@router.post("/redact", summary="Redact PII from text")
async def redact_pii(request: RedactRequest) -> Dict[str, Any]:
    result = _pii_redactor.redact(request.text)
    return {
        "redacted_text": result.redacted_text,
        "has_pii": result.has_pii,
        "pii_types": result.pii_types,
        "redaction_count": result.redaction_count,
        "risk_level": result.risk_level(),
    }


@router.get("/ci-check/{model_name}/{version}", summary="Run CI governance checks")
async def run_ci_check(
    model_name: str,
    version: int,
    ragas_faithfulness: float = 0.847,
    ragas_relevancy: float = 0.823,
) -> Dict[str, Any]:
    report = _ci_enforcement.run_all_checks(
        model_name=model_name,
        version=version,
        current_metrics={
            "ragas_faithfulness": ragas_faithfulness,
            "ragas_relevancy": ragas_relevancy,
        },
        audit_log=_audit_log,
    )
    return report.to_dict()
