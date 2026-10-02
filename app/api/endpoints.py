"""API routes for filing review and conversation history."""

import logging
from typing import Any
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, ToolInfo, SessionHistoryResponse
from app.agent.agent_service import execute_agent_query
from app.agent.memory import get_formatted_history, clear_session_history
from app.tools.base import get_tool_definitions, get_all_tools
from app.filing_service import load_filings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


@router.get("/filings", response_model=list[str])
def list_filings():
    """List record IDs without exposing text or actual labels."""
    return load_filings().index.tolist()


@router.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Ask Gemini to review filings using the local tool and session history."""
    try:
        return execute_agent_query(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail="Review failed. Check server logs and provider settings.") from exc


@router.get("/tools", response_model=list[ToolInfo])
def list_tools():
    return get_tool_definitions()


@router.post("/tools/execute/{tool_name}")
def direct_tool_execute(tool_name: str, payload: dict[str, Any]):
    """Test a registered tool locally without making an LLM request."""
    tools = {tool.name: tool for tool in get_all_tools()}
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
    try:
        result = tools[tool_name].invoke(payload)
        return {"tool": tool_name, "input": payload, "result": str(result), "status": "success"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Filing tool execution failed")
        raise HTTPException(status_code=500, detail="Filing review failed. Check the dataset and trained model.") from exc


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
def get_history(session_id: str):
    messages = get_formatted_history(session_id)
    return SessionHistoryResponse(session_id=session_id, message_count=len(messages), messages=messages)


@router.delete("/history/{session_id}")
def clear_history(session_id: str):
    return {"session_id": session_id, "cleared": clear_session_history(session_id)}
