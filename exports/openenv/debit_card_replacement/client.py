"""Client for OpenEnv environment: debit_card_replacement"""

from openenv.core import EnvClient, StepResult

from .models import VoiceAction, VoiceObservation, VoiceState


class DebitCardReplacementEnv(EnvClient[VoiceAction, VoiceObservation, VoiceState]):
    """OpenEnv client for the debit_card_replacement voice environment."""

    def _step_payload(self, action: VoiceAction) -> dict:
        return {"content": action.content, "tool_calls": action.tool_calls}

    def _parse_result(self, payload: dict) -> StepResult[VoiceObservation]:
        obs = VoiceObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> VoiceState:
        return VoiceState(**payload)