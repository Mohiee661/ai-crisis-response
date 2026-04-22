def send_alert(action: str) -> None:
    if action == "ALERT_FIRE_STATION":
        print("🚒 Fire Alert Sent")
    elif action == "ALERT_AMBULANCE":
        print("🚑 Ambulance Alert Sent")
    elif action == "ALERT_BOTH":
        print("🚒🚑 Fire + Ambulance Alert Sent")
