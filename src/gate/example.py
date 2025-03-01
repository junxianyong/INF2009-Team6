def face_detected():
    """
    TODO: This function is a infinte loop that waits for a face to be detected
    """
    while True:
        if input("Face detected? (y/n)").lower() == "y":
            return True


def open_gate(id):
    """
    TODO: This function will open the gate
    """
    print(f"Gate {id} opened")


def face_verified():
    """
    TODO: This function is a infinte loop that waits for a face to be verified (after few tries?),
    it should return personnel ID if the face is verified, None otherwise
    """
    while True:
        x = input("Face verified? (y/n)")
        if x.lower() == "y":
            return "personnel_id"
        else:
            return None


def face_verified_with_id(personnel_id):
    """
    TODO: This function is a infinte loop that waits for a face to be verified with personnel ID,
    it should return True if the face is verified
    """
    while True:
        x = input(f"Face verified with personnel ID {personnel_id}? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False


def voice_verified(personnel_id):
    """
    TODO: This function is a infinte loop that waits for a voice to be verified with personnel ID,
    it should return True if the face is verified
    """
    while True:
        x = input(f"Voice verified with personnel ID {personnel_id}? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False


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


def capture_intruder():
    """
    TODO: This function will capture the intruder and return the image buffer
    """
    print("Intruder captured")
    return "image_buffer"


def alert_buzzer_and_led():
    """
    TODO: This function will alert the buzzer and LED (Beep and Blink once and delay for 1 second)
    """
    time.sleep(1)
    print("Buzzer and LED alerted")
