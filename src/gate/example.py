import time

def open_gate(id):
    """
    TODO: This function will open the gate
    """
    print(f"Gate {id} opened")


def personnel_passed():
    """
    TODO: This function is a infinte loop that waits for the personnel to pass through the
    gate, it should return True if the personnel has passed
    """
    while True:
        x = input("Personnel passed? (y/n)")
        if x.lower() == "y":
            return True


def close_gate(id):
    """
    TODO: This function will close the gate
    """
    print(f"Gate {id} closed")


def mantrap_scan():
    """
    TODO: This function is a infinte loop that waits for the mantrap to scan the personnel,
    if there is multiple personnel, it should return False, True otherwise
    """
    while True:
        x = input("Only one personnel in mantrap? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False

def alert_buzzer_and_led():
    """
    TODO: This function will alert the buzzer and LED (Beep and Blink once and delay for 1 second)
    """
    time.sleep(1)
    print("Buzzer and LED alerted")
