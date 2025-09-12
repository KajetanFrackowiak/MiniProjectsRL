#!/usr/bin/env python3
"""
Simple test to check if the issue is with environment creation or rendering.
"""

import gymnasium as gym
import os


def test_display_setup():
    """Check if display is properly configured."""
    print("Checking display configuration...")
    display = os.environ.get("DISPLAY")
    if display:
        print(f"DISPLAY environment variable: {display}")
    else:
        print("DISPLAY environment variable not set")

    # Check if we're in a graphical session
    if os.environ.get("XDG_SESSION_TYPE"):
        print(f"Session type: {os.environ.get('XDG_SESSION_TYPE')}")

    if os.environ.get("WAYLAND_DISPLAY"):
        print(f"Wayland display: {os.environ.get('WAYLAND_DISPLAY')}")


def test_simple_env():
    """Test very basic environment functionality."""
    print("\nTesting basic environment creation (no rendering)...")
    try:
        env = gym.make("CartPole-v1")
        print("✅ Environment created successfully")

        obs, _ = env.reset()
        print("✅ Environment reset successfully")

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"✅ Step taken successfully, reward: {reward}")

        env.close()
        print("✅ Environment closed successfully")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_render_env_creation():
    """Test environment creation with render mode."""
    print("\nTesting environment creation with render_mode='human'...")
    try:
        env = gym.make("CartPole-v1", render_mode="human")
        print("✅ Render environment created successfully")
        env.close()
        return True

    except Exception as e:
        print(f"❌ Error creating render environment: {e}")
        return False


def main():
    print("Quick Render Diagnostic")
    print("=" * 30)

    test_display_setup()

    basic_success = test_simple_env()
    if not basic_success:
        print("\n❌ Basic environment functionality failed!")
        return

    render_creation_success = test_render_env_creation()
    if not render_creation_success:
        print("\n❌ Render environment creation failed!")
        print("This suggests a graphics/display issue.")
        return

    print("\n✅ Basic tests passed. The issue might be with the render() call itself.")
    print(
        "Try running your script with render=False to confirm it works without rendering."
    )


if __name__ == "__main__":
    main()
