from src.overworld.skills.base import SkillStatus
from scripts.demo_overworld_corridor import run_corridor_demo

def test_corridor_demo_reaches_goal():
    actions, status = run_corridor_demo(length=4)
    assert status == SkillStatus.SUCCEEDED
    assert any(action.get("kind") == "button" for action in actions)
