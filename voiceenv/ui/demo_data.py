"""
Seed the ratings store with realistic demo conversations so the UI
is not empty when someone visits for the first time.
"""

from __future__ import annotations

from voiceenv.core.human_ratings import RatingStore, RunForRating


DEMO_RUNS: list[dict] = [
    {
        "run_id": "demo_healthcare_001",
        "environment_name": "Healthcare Triage - Chest Pain",
        "transcript": [
            {"role": "user", "content": "Hi, I'm calling because I've been having some chest pain for the last hour or so."},
            {"role": "agent", "content": "I'm sorry to hear that. Let me help you. Can you describe the pain — is it sharp, dull, or more of a pressure feeling?"},
            {"role": "user", "content": "It's more like a tightness... like something is sitting on my chest."},
            {"role": "agent", "content": "Thank you for describing that. A few quick questions: Are you having any trouble breathing? Any pain going down your left arm or up into your jaw?"},
            {"role": "user", "content": "I'm a little short of breath actually. No arm or jaw pain though."},
            {"role": "agent", "content": "Okay. Given chest tightness with shortness of breath, I want to make sure you're safe. Do you have any history of heart problems, high blood pressure, or diabetes?"},
            {"role": "user", "content": "I have high blood pressure, I take medication for it."},
            {"role": "agent", "content": "Thank you. Based on what you're telling me — chest tightness, shortness of breath, and a history of high blood pressure — I'm going to recommend you call 911 or have someone drive you to the nearest emergency room right away. This could be something that needs immediate medical attention. Would you like me to stay on the line while you call?"},
            {"role": "user", "content": "Oh... okay, that's scary. Yes, please stay on the line."},
            {"role": "agent", "content": "I understand this is frightening, and you're doing the right thing by calling. You're going to be okay. Let's get you the help you need. I'll stay right here with you."},
        ],
        "criteria_to_rate": [
            {"name": "urgency_assessment", "description": "Did the agent correctly identify the urgency level of the situation?"},
            {"name": "calm_reassuring_tone", "description": "Did the agent speak calmly and reassuringly, without causing panic?"},
            {"name": "correct_triage", "description": "Did the agent recommend the right course of action (call 911)?"},
            {"name": "information_gathering", "description": "Did the agent ask the right questions to assess the situation?"},
            {"name": "empathy", "description": "Did the agent show empathy and compassion for the caller?"},
        ],
        "llm_scores": {
            "urgency_assessment": 0.92,
            "calm_reassuring_tone": 0.85,
            "correct_triage": 0.95,
            "information_gathering": 0.88,
            "empathy": 0.80,
        },
    },
    {
        "run_id": "demo_restaurant_002",
        "environment_name": "Restaurant Reservation",
        "transcript": [
            {"role": "user", "content": "Hello, I'd like to make a reservation for tonight."},
            {"role": "agent", "content": "Of course! I'd be happy to help. How many people will be dining tonight?"},
            {"role": "user", "content": "Four people. And we need a high chair for a toddler."},
            {"role": "agent", "content": "Four guests plus a high chair, got it. What time works best for you this evening?"},
            {"role": "user", "content": "Around 7pm if possible."},
            {"role": "agent", "content": "Let me check... I have availability at 7pm and 7:30pm. The 7pm slot is at a nice corner booth which would be great for your family. Would you prefer that?"},
            {"role": "user", "content": "The corner booth sounds perfect!"},
            {"role": "agent", "content": "Wonderful! I have you down for 4 guests at 7pm, corner booth, with a high chair. May I have a name for the reservation?"},
            {"role": "user", "content": "It's under Patel. And one of us has a peanut allergy — is that something the kitchen can handle?"},
            {"role": "agent", "content": "Absolutely, Mr. Patel. I've noted the peanut allergy on your reservation. Our chef will prepare your dishes in a peanut-free area. You're all set for tonight at 7pm. See you then!"},
        ],
        "criteria_to_rate": [
            {"name": "task_completion", "description": "Did the agent successfully complete the reservation with all necessary details?"},
            {"name": "friendliness", "description": "Was the agent warm, friendly, and welcoming?"},
            {"name": "accommodation", "description": "Did the agent handle special requests (high chair, allergy) well?"},
            {"name": "efficiency", "description": "Did the agent complete the task without unnecessary back-and-forth?"},
        ],
        "llm_scores": {
            "task_completion": 0.95,
            "friendliness": 0.90,
            "accommodation": 0.93,
            "efficiency": 0.87,
        },
    },
    {
        "run_id": "demo_tech_support_003",
        "environment_name": "Tech Support - WiFi Troubleshooting",
        "transcript": [
            {"role": "user", "content": "My internet isn't working and I have a work meeting in 20 minutes!"},
            {"role": "agent", "content": "I understand the urgency. Let's get this sorted quickly. First, can you tell me — are any of the lights on your router blinking or off?"},
            {"role": "user", "content": "Umm... the power light is on but the internet light is red."},
            {"role": "agent", "content": "Okay, a red internet light usually means the connection to your provider is down. Let's try a quick restart. Can you unplug the power cable from your router, wait 10 seconds, then plug it back in?"},
            {"role": "user", "content": "Okay, I did that. It's restarting now... the lights are blinking."},
            {"role": "agent", "content": "Great, give it about a minute to fully boot up. While we wait — is your computer connected via WiFi or an ethernet cable?"},
            {"role": "user", "content": "WiFi. Oh wait, the internet light just turned green!"},
            {"role": "agent", "content": "Excellent! Try opening a webpage now. Is it loading?"},
            {"role": "user", "content": "Yes! It's working! Thank you so much, you saved my meeting!"},
            {"role": "agent", "content": "Happy to help! If it drops again, give us a call and we can run deeper diagnostics. Good luck with your meeting!"},
        ],
        "criteria_to_rate": [
            {"name": "problem_resolution", "description": "Did the agent successfully resolve the technical issue?"},
            {"name": "clear_instructions", "description": "Were the troubleshooting steps easy to understand and follow?"},
            {"name": "urgency_awareness", "description": "Did the agent acknowledge and respond to the time pressure?"},
            {"name": "technical_accuracy", "description": "Were the agent's technical explanations correct?"},
        ],
        "llm_scores": {
            "problem_resolution": 0.95,
            "clear_instructions": 0.90,
            "urgency_awareness": 0.82,
            "technical_accuracy": 0.88,
        },
    },
]


def seed_demo_data(ratings_dir: str = "ratings") -> int:
    """Seed the rating store with demo runs. Returns number of runs seeded."""
    store = RatingStore(ratings_dir)
    seeded = 0

    for run_data in DEMO_RUNS:
        run = RunForRating(
            run_id=run_data["run_id"],
            environment_name=run_data["environment_name"],
            transcript=run_data["transcript"],
            criteria_to_rate=run_data["criteria_to_rate"],
            llm_scores=run_data["llm_scores"],
        )
        store.save_run_for_rating(run)
        seeded += 1

    return seeded
