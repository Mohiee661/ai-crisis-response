from social_agent import fetch_social_signals


def test_fetch_social_fire_signal() -> None:
    signal = fetch_social_signals("fire reported in mall food court", location="mall")

    assert signal["type"] == "FIRE"
    assert signal["source"] == "social"
    assert signal["confidence"] >= 0.70
