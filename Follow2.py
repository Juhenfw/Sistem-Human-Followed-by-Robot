def control_loop_callback(self):
    """🎯 MAIN CONTROL LOOP - HANYA AKTIF SAAT MODE OTOMATIS"""
    
    # Mode checking
    if not self.mode_active or self.current_mode != 'otomatis':
        return
    
    start_time = time.time()
    
    try:
        # DEBUG: Log status setiap 2 detik
        current_time = time.time()
        if current_time - getattr(self, 'last_debug_log', 0) > 2.0:
            print(f"🤖 AUTONOMOUS MODE ACTIVE - Looking for UWB data...")
            self.last_debug_log = current_time
        
        # Get UWB data dengan logging
        uwb_distances = self.get_uwb_data()
        
        if uwb_distances:
            print(f"📡 UWB DATA RECEIVED: A0={uwb_distances['A0']:.1f}cm, A1={uwb_distances['A1']:.1f}cm, A2={uwb_distances['A2']:.1f}cm")
            
            # Process control
            self.controller.process_control_ultra_fast(
                uwb_distances, 
                self.lidar, 
                self.ultrasonic_manager
            )
            
            # Update target info
            target_direction, target_distance = self.uwb_tracker.estimate_target_direction(uwb_distances)
            self.lidar.set_target_info(target_direction, target_distance)
            
        else:
            # DEBUG: UWB tidak ada data
            if current_time - getattr(self, 'last_uwb_warning', 0) > 5.0:
                print("⚠️ NO UWB DATA RECEIVED - Check UWB transmitter")
                self.last_uwb_warning = current_time
            
    except Exception as e:
        print(f"❌ Control loop error: {e}")
        import traceback
        traceback.print_exc()

def get_uwb_data(self):
    """Get UWB data from socket dengan enhanced debugging"""
    if not self.uwb_socket:
        print("❌ UWB socket not initialized")
        return None
    
    try:
        data, addr = self.uwb_socket.recvfrom(2048)
        received_data = data.decode('utf-8').strip()
        
        # DEBUG: Log raw data
        print(f"📡 RAW UWB: {received_data} from {addr}")
        
        # Parse UWB data
        parts = received_data.split(',')
        if len(parts) >= 3:
            distances = {
                'A0': float(parts[0]),
                'A1': float(parts[1]),
                'A2': float(parts[2])
            }
            
            # Apply bias correction
            corrected_distances = self.uwb_tracker.apply_bias_correction(distances)
            return corrected_distances
        else:
            print(f"❌ Invalid UWB data format: {received_data}")
            
    except socket.timeout:
        # Normal timeout - tidak perlu log setiap saat
        pass
    except Exception as e:
        print(f"❌ UWB data error: {e}")
    
    return None
