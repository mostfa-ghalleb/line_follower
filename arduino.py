import serial
import serial.tools.list_ports
from typing import Optional

def connect_to_arduino() -> Optional[serial.Serial]:
    """Establish a serial connection to an Arduino device.
    
    Returns:
        Serial object if connected, None otherwise.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'arduino' in port.description.lower():
            try:
                ser = serial.Serial(port.device, 9600, timeout=1)
                print(f"Connected to Arduino on {port.device}")
                return ser
            except Exception as e:
                print(f"Failed to connect to {port.device}: {e}")
    print("No Arduino found!")
    return None

def send_to_arduino(ser: Optional[serial.Serial], message: str) -> bool:
    """Send a message to the Arduino.
    
    Args:
        ser: Serial connection to Arduino.
        message: Message to send.
    
    Returns:
        True if sent successfully, False otherwise.
    """
    if ser is None:
        print("No Arduino connection available")
        return False
    try:
        if not message.endswith('\n'):
            message += '\n'
        ser.write(message.encode('utf-8'))
        return True
    except Exception as e:
        print(f"Error sending to Arduino: {e}")
        return False