def send_alert(action: str) -> None:
    if action == "ALERT_FIRE_STATION":
        print("[ALERT] Fire station alert sent")
    elif action == "ALERT_AMBULANCE":
        print("[ALERT] Ambulance alert sent")
    elif action == "ALERT_BOTH":
        print("[ALERT] Fire station + ambulance alert sent")
