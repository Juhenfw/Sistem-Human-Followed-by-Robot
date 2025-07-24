# Gunakan rpi-lgpio sebagai pengganti RPi.GPIO
import RPi.GPIO as GPIO  # Ini akan menggunakan rpi-lgpio
import time
import threading

# Konfigurasi GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Definisi pin untuk 3 sensor
SENSORS = {
    'sensor1': {'trig': 18, 'echo': 24},
    'sensor2': {'trig': 23, 'echo': 25},
    'sensor3': {'trig': 12, 'echo': 16}
}

class UltrasonicSensor:
    def __init__(self, name, trig_pin, echo_pin):
        self.name = name
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        
        try:
            # Setup GPIO pins
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, False)
            
            # Waktu untuk sensor settle
            time.sleep(0.1)
            print(f"✓ {name} initialized (Trig: GPIO{trig_pin}, Echo: GPIO{echo_pin})")
            
        except Exception as e:
            print(f"✗ Error initializing {name}: {e}")
            raise
    

    def measure_distance(self):
        """Mengukur jarak dalam cm"""
        try:
            # Kirim trigger pulse (10us untuk JSN-SR04T)
            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)  # 10 microseconds (lebih stabil untuk RPi 5)
            GPIO.output(self.trig_pin, False)
            
            # Tunggu echo response dengan timeout yang lebih pendek
            timeout_start = time.time()
            timeout_duration = 0.03  # 30ms timeout
            
            # Tunggu echo pin HIGH
            pulse_start = timeout_start
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if (pulse_start - timeout_start) > timeout_duration:
                    return -1
            
            # Tunggu echo pin LOW
            pulse_end = pulse_start
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if (pulse_end - timeout_start) > timeout_duration:
                    return -1
            
            # Hitung jarak
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Kecepatan suara 343 m/s
            
            # Filter hasil yang masuk akal
            if 2 <= distance <= 400:
                return round(distance, 1)
            else:
                return -1
                
        except Exception as e:
            print(f"Error measuring {self.name}: {e}")
            return -1

def read_sensor(sensor):
    """Fungsi untuk membaca sensor secara berulang"""
    while True:
        distance = sensor.measure_distance()
        current_time = time.strftime("%H:%M:%S")
        
        if distance > 0:
            print(f"[{current_time}] {sensor.name}: {distance} cm")
        else:
            print(f"[{current_time}] {sensor.name}: No signal")
        
        time.sleep(0.1)  # Delay lebih pendek untuk responsivitas

def test_gpio_availability():
    """Test apakah GPIO bisa digunakan"""
    try:
        # Test sederhana
        test_pin = 18
        GPIO.setup(test_pin, GPIO.OUT)
        GPIO.output(test_pin, False)
        print("✓ GPIO test successful")
        return True
    except Exception as e:
        print(f"✗ GPIO test failed: {e}")
        return False

def main():
    try:
        print("=== JSN-SR04T Ultrasonic Sensor for RPi 5 ===")
        print("Using rpi-lgpio library")
        print()
        
        # Test GPIO terlebih dahulu
        if not test_gpio_availability():
            print("GPIO not available. Please check installation and permissions.")
            return
        
        # Inisialisasi sensor
        sensors = []
        print("Initializing sensors...")
        
        for name, pins in SENSORS.items():
            try:
                sensor = UltrasonicSensor(name, pins['trig'], pins['echo'])
                sensors.append(sensor)
            except Exception as e:
                print(f"Failed to initialize {name}: {e}")
                continue
        
        if not sensors:
            print("No sensors could be initialized!")
            return
        
        print(f"\n✓ {len(sensors)} sensors initialized successfully!")
        print("Starting measurements...")
        print("Press Ctrl+C to exit\n")
        
        # Pembacaan berurutan
        while True:
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            
            for sensor in sensors:
                distance = sensor.measure_distance()
                if distance > 0:
                    print(f"{sensor.name:>8}: {distance:>6.1f} cm")
                else:
                    print(f"{sensor.name:>8}: {'No signal':>10}")
                
                time.sleep(0.01)  # Small delay between sensors
            
            time.sleep(0.2)  # Delay antar cycle
            
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except:
            pass

def main_parallel():
    try:
        print("=== Parallel Reading Mode ===")
        
        if not test_gpio_availability():
            print("GPIO not available!")
            return
        
        # Inisialisasi sensor dan threads
        sensors = []
        threads = []
        
        print("Initializing sensors...")
        for name, pins in SENSORS.items():
            try:
                sensor = UltrasonicSensor(name, pins['trig'], pins['echo'])
                sensors.append(sensor)
                
                # Buat thread untuk setiap sensor
                thread = threading.Thread(target=read_sensor, args=(sensor,))
                thread.daemon = True
                threads.append(thread)
                
            except Exception as e:
                print(f"Skipping {name}: {e}")
                continue
        
        if not sensors:
            print("No sensors available!")
            return
        
        # Mulai semua thread dengan delay
        print(f"\n✓ Starting {len(sensors)} sensors in parallel...")
        for i, thread in enumerate(threads):
            thread.start()
            time.sleep(0.1)  # Stagger start untuk menghindari interference
        
        print("Parallel reading started...")
        print("Press Ctrl+C to exit\n")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except:
            pass

if __name__ == "__main__":
    try:
        print("Choose reading method:")
        print("1. Sequential reading (recommended)")
        print("2. Parallel reading")
        
        choice = input("Enter choice (1/2): ").strip()
        print()
        
        if choice == "2":
            main_parallel()
        else:
            main()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Startup error: {e}")
    finally:
        try:
            GPIO.cleanup()
        except:
            pass
