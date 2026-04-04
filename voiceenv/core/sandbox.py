"""
Sandbox executor for tool calls inside a voice environment.

Tools are simulated — no real CRM/calendar/email calls happen. The sandbox
maintains world state and returns realistic mock responses. Tool failure
rates and latencies can be configured per-tool for stress testing.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from voiceenv.core.schema import ToolDefinition, WorldState


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    success: bool
    latency_ms: int
    timestamp: float
    state_before: dict[str, Any]
    state_after: dict[str, Any]


class Sandbox:
    """
    Executes tool calls against simulated backends and tracks world state mutations.
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        initial_state: WorldState,
    ):
        self.tool_defs = {t.name: t for t in tools}
        self.state = copy.deepcopy(initial_state.fields)
        self.call_history: list[ToolCallRecord] = []

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling compatible tool schemas."""
        schemas = []
        for tool in self.tool_defs.values():
            params = {}
            required = []
            for p in tool.parameters:
                param_schema: dict[str, Any] = {"type": p.type, "description": p.description}
                if p.enum:
                    param_schema["enum"] = p.enum
                if p.default is not None:
                    param_schema["default"] = p.default
                params[p.name] = param_schema
                if p.required:
                    required.append(p.name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            })
        return schemas

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return the result."""
        tool = self.tool_defs.get(tool_name)
        if not tool:
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result={"error": f"Unknown tool: {tool_name}"},
                success=False,
                latency_ms=0,
                timestamp=time.time(),
                state_before=copy.deepcopy(self.state),
                state_after=copy.deepcopy(self.state),
            )
            self.call_history.append(record)
            return record.result

        state_before = copy.deepcopy(self.state)

        if tool.latency_ms > 0:
            time.sleep(tool.latency_ms / 1000.0)

        # Simulate failure based on success_rate
        if random.random() > tool.success_rate:
            result = {"error": f"Tool '{tool_name}' failed (simulated failure)", "success": False}
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                success=False,
                latency_ms=tool.latency_ms,
                timestamp=time.time(),
                state_before=state_before,
                state_after=copy.deepcopy(self.state),
            )
            self.call_history.append(record)
            return result

        # Execute the tool (simulated)
        result = self._simulate_tool(tool_name, arguments)

        # Apply side effects
        if tool.side_effects:
            for key, value in tool.side_effects.items():
                self.state[key] = value

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=True,
            latency_ms=tool.latency_ms,
            timestamp=time.time(),
            state_before=state_before,
            state_after=copy.deepcopy(self.state),
        )
        self.call_history.append(record)
        return result

    def _simulate_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Generate plausible mock responses for common tool types."""
        name_lower = tool_name.lower()

        if "book" in name_lower or "schedule" in name_lower or "calendar" in name_lower:
            slot = args.get("time", args.get("slot", args.get("datetime", "2026-04-10T10:00:00")))
            self.state["meeting_booked"] = True
            self.state["meeting_time"] = slot
            return {
                "success": True,
                "meeting_id": f"mtg_{random.randint(10000, 99999)}",
                "scheduled_time": slot,
                "confirmation": f"Meeting confirmed for {slot}",
            }

        if "crm" in name_lower or "update" in name_lower or "log" in name_lower:
            for k, v in args.items():
                self.state[f"crm_{k}"] = v
            return {"success": True, "message": "CRM updated", "fields_updated": list(args.keys())}

        if "email" in name_lower or "send" in name_lower:
            return {
                "success": True,
                "message_id": f"msg_{random.randint(10000, 99999)}",
                "sent_to": args.get("to", args.get("recipient", "user@example.com")),
            }

        if "search" in name_lower or "lookup" in name_lower or "knowledge" in name_lower:
            query = args.get("query", args.get("q", ""))
            return {
                "success": True,
                "results": [
                    {"title": f"Result for: {query}", "snippet": "Relevant information found.", "score": 0.92}
                ],
            }

        if "transfer" in name_lower or "escalate" in name_lower:
            self.state["escalated"] = True
            return {
                "success": True,
                "transferred_to": args.get("department", args.get("agent", "supervisor")),
            }

        if "payment" in name_lower or "charge" in name_lower:
            amount = args.get("amount", 0)
            self.state["payment_collected"] = True
            self.state["payment_amount"] = amount
            return {"success": True, "transaction_id": f"txn_{random.randint(10000, 99999)}", "amount": amount}

        if "ticket" in name_lower or "create" in name_lower:
            return {
                "success": True,
                "ticket_id": f"TKT-{random.randint(10000, 99999)}",
                "status": "created",
            }

        # Generic fallback
        return {"success": True, "result": f"Executed {tool_name}", "args": args}

    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def get_call_log(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": r.tool_name,
                "arguments": r.arguments,
                "result": r.result,
                "success": r.success,
                "latency_ms": r.latency_ms,
            }
            for r in self.call_history
        ]
