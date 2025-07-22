import RPi.GPIO as GPIO
import time
import threading

# Konfigurasi GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Definisi pin untuk 3 sensor
SENSORS = {
    'sensor1': {'trig': 18, 'echo': 24},
    'sensor2': {'trig': 23, 'echo': 25},
    'sensor3': {'trig': 8, 'echo': 7}
}

class UltrasonicSensor:
    def __init__(self, name, trig_pin, echo_pin):
        self.name = name
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        
        # Setup GPIO pins
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, False)
        
        # Waktu untuk sensor settle
        time.sleep(0.1)
    
    def measure_distance(self):
        """Mengukur jarak dalam cm"""
        # Kirim trigger pulse (20us untuk JSN-SR04T)[26]
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00002)  # 20 microseconds
        GPIO.output(self.trig_pin, False)
        
        # Tunggu echo response
        timeout = time.time() + 0.1  # 100ms timeout
        
        # Tunggu echo pin HIGH
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return -1
        
        # Tunggu echo pin LOW
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return -1
        
        # Hitung jarak
        try:
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Kecepatan suara 343 m/s
            return round(distance, 1)
        except:
            return -1

def read_sensor(sensor):
    """Fungsi untuk membaca sensor secara berulang"""
    while True:
        distance = sensor.measure_distance()
        if distance > 0:
            print(f"{sensor.name}: {distance} cm")
        else:
            print(f"{sensor.name}: Error reading")
        time.sleep(1)

def main():
    try:
        # Inisialisasi sensor
        sensors = []
        for name, pins in SENSORS.items():
            sensor = UltrasonicSensor(name, pins['trig'], pins['echo'])
            sensors.append(sensor)
        
        print("Memulai pembacaan 3 sensor JSN-SR04T...")
        print("Tekan Ctrl+C untuk keluar")
        
        # Metode 1: Pembacaan berurutan
        while True:
            print("\n--- Pembacaan Sensor ---")
            for sensor in sensors:
                distance = sensor.measure_distance()
                if distance > 0:
                    print(f"{sensor.name}: {distance} cm")
                else:
                    print(f"{sensor.name}: Error")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user")
    finally:
        GPIO.cleanup()

# Fungsi alternatif untuk pembacaan paralel
def main_parallel():
    try:
        # Inisialisasi sensor
        sensors = []
        threads = []
        
        for name, pins in SENSORS.items():
            sensor = UltrasonicSensor(name, pins['trig'], pins['echo'])
            sensors.append(sensor)
            
            # Buat thread untuk setiap sensor
            thread = threading.Thread(target=read_sensor, args=(sensor,))
            thread.daemon = True
            threads.append(thread)
        
        # Mulai semua thread
        for thread in threads:
            thread.start()
        
        print("Pembacaan paralel dimulai...")
        print("Tekan Ctrl+C untuk keluar")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    # Pilih salah satu metode
    main()  # Pembacaan berurutan
    # main_parallel()  # Pembacaan paralel
