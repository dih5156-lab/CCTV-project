"""Authenticated, binding-scoped storage status; MQTT owns all record writes."""

import secrets
import sqlite3

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .monitoring_contract import MonitoringBinding, MonitoringContractError


def create_monitoring_api(store, binding, token, *, receiver=None, lifespan=None):
    if (
        not isinstance(token, str)
        or len(token) < 32
        or not token.isascii()
        or any(c.isspace() for c in token)
    ):
        raise ValueError(
            "A nonempty ASCII API token of at least 32 characters is required"
        )
    if store.consumer_id != binding.consumer_id:
        raise MonitoringContractError("binding_mismatch")
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(request: Request):
        supplied = request.headers.get("authorization", "").encode("utf-8")
        if not secrets.compare_digest(supplied, ("Bearer " + token).encode()):
            raise HTTPException(401, detail="unauthorized")

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(
            {"error": {"code": error.detail}}, status_code=error.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"error": {"code": "invalid_request"}}, status_code=422)

    @app.get("/v1/health", dependencies=[Depends(authorize)])
    def health():
        ready = receiver is None or receiver.is_ready()
        return JSONResponse(
            {
                "status": "ready" if ready else "waiting_for_mqtt",
                "binding": binding.as_dict(),
            },
            status_code=200 if ready else 503,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/deliveries/{delivery_id}", dependencies=[Depends(authorize)])
    def delivery_status(
        delivery_id: str, origin_id: str, consumer_id: str, destination: str
    ):
        try:
            requested = MonitoringBinding(origin_id, consumer_id, destination)
            if requested != binding:
                raise MonitoringContractError("binding_mismatch")
        except ValueError:
            raise HTTPException(403, detail="binding_mismatch") from None
        if not delivery_id.strip() or len(delivery_id) > 512:
            raise HTTPException(422, detail="invalid_request")
        try:
            ack = store.get_delivery_status(binding, delivery_id)
        except MonitoringContractError:
            raise HTTPException(403, detail="binding_mismatch") from None
        except (sqlite3.Error, OSError):
            raise HTTPException(503, detail="storage_unavailable") from None
        if ack is None:
            return JSONResponse(
                {
                    "status": "not_found",
                    "binding": binding.as_dict(),
                    "delivery_id": delivery_id,
                },
                status_code=404,
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(ack, headers={"Cache-Control": "no-store"})

    return app
