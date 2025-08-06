def execute_obstacle_avoidance_movement(self, strategy, base_left_speed, base_right_speed):
    """
    Eksekusi pergerakan obstacle avoidance berdasarkan strategi yang diberikan.
    
    Parameters:
        strategy (dict): strategi penghindaran dari evaluate_obstacle_avoidance_strategy
        base_left_speed (float): kecepatan roda kiri sebelum adjustment
        base_right_speed (float): kecepatan roda kanan sebelum adjustment
    
    Returns:
        tuple: (adjusted_left_speed, adjusted_right_speed)
    """
    if not strategy or 'left_speed_factor' not in strategy or 'right_speed_factor' not in strategy:
        # Jika strategi tidak valid, kembalikan kecepatan dasar
        return base_left_speed, base_right_speed

    left_factor = strategy.get('left_speed_factor', 1.0)
    right_factor = strategy.get('right_speed_factor', 1.0)

    adjusted_left_speed = base_left_speed * left_factor
    adjusted_right_speed = base_right_speed * right_factor

    print(f"Executing avoidance: {strategy.get('description', 'No description')}")
    print(f"Speed factors applied - Left: {left_factor}, Right: {right_factor}")
    print(f"Speeds adjusted from L={base_left_speed} R={base_right_speed} to L={adjusted_left_speed} R={adjusted_right_speed}")

    return adjusted_left_speed, adjusted_right_speed

def generate_escape_strategy(self, area_name, area_data):
    """Generate escape strategy berdasarkan area yang dipilih"""
    
    if area_data['clearance'] > 500:  # 50cm clearance
        speed_factor = 0.8
        priority = 'MEDIUM'
    elif area_data['clearance'] > 300:  # 30cm clearance
        speed_factor = 0.6
        priority = 'HIGH'
    else:
        speed_factor = 0.4
        priority = 'HIGH'
    
    if area_name == 'sharp_left':
        return {
            'action': 'sharp_left_escape',
            'description': f'Sharp left escape - clearance {area_data["clearance"]:.0f}mm',
            'left_speed_factor': -speed_factor,
            'right_speed_factor': -speed_factor,
            'priority': priority
        }
    elif area_name == 'sharp_right':
        return {
            'action': 'sharp_right_escape',
            'description': f'Sharp right escape - clearance {area_data["clearance"]:.0f}mm',
            'left_speed_factor': speed_factor,
            'right_speed_factor': speed_factor,
            'priority': priority
        }
    elif area_name == 'left_diagonal':
        return {
            'action': 'diagonal_left_escape',
            'description': f'Diagonal left escape - clearance {area_data["clearance"]:.0f}mm',
            'left_speed_factor': speed_factor * 0.7,
            'right_speed_factor': speed_factor,
            'priority': priority
        }
    elif area_name == 'right_diagonal':
        return {
            'action': 'diagonal_right_escape',
            'description': f'Diagonal right escape - clearance {area_data["clearance"]:.0f}mm',
            'left_speed_factor': speed_factor,
            'right_speed_factor': speed_factor * 0.7,
            'priority': priority
        }
    elif area_name == 'reverse':
        return {
            'action': 'reverse_escape',
            'description': f'Reverse escape - clearance {area_data["clearance"]:.0f}mm',
            'left_speed_factor': -speed_factor * 0.5,
            'right_speed_factor': speed_factor * 0.5,
            'priority': 'HIGH'
        }
    else:
        # Default emergency strategy
        return {
            'action': 'emergency_stop',
            'description': 'Emergency stop - no safe escape route',
            'left_speed_factor': 0,
            'right_speed_factor': 0,
            'priority': 'HIGH'
        }

def get_emergency_fallback_strategy(self):
    """Emergency fallback strategy saat LIDAR tidak tersedia"""
    return {
        'action': 'emergency_reverse',
        'description': 'Emergency reverse - LIDAR unavailable',
        'left_speed_factor': -0.3,
        'right_speed_factor': 0.3,
        'priority': 'HIGH'
    }
