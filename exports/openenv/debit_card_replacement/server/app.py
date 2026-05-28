"""FastAPI app for OpenEnv environment: debit_card_replacement"""

from openenv.core.env_server import create_fastapi_app

from ..models import VoiceAction, VoiceObservation
from .environment import VoiceAgentEnvironment

env = VoiceAgentEnvironment()
app = create_fastapi_app(env, VoiceAction, VoiceObservation)