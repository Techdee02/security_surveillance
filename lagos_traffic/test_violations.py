#!/usr/bin/env python3
"""
Test script to create sample violations for testing the review UI
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from modules.violations import (
    get_violations_db, ViolationType
)
import random


def create_sample_violations():
    """Create sample violations for testing"""
    db = get_violations_db()
    
    # Sample data
    vehicle_types = ['okada', 'danfo', 'private_car', 'keke_napep', 'bus', 'truck']
    
    sample_violations = [
        {
            'violation_type': ViolationType.NO_HELMET,
            'vehicle_type': 'okada',
            'confidence': 0.92,
            'plate_number': 'LAG-123-AB',
            'zone_id': 'ZONE-A1'
        },
        {
            'violation_type': ViolationType.NO_HELMET,
            'vehicle_type': 'okada',
            'confidence': 0.87,
            'plate_number': 'KJA-456-CD',
            'zone_id': 'ZONE-B2'
        },
        {
            'violation_type': ViolationType.STATIONARY_OBSTRUCTION,
            'vehicle_type': 'danfo',
            'confidence': 0.95,
            'plate_number': 'APP-789-EF',
            'zone_id': 'ZONE-A1'
        },
        {
            'violation_type': ViolationType.WRONG_WAY,
            'vehicle_type': 'okada',
            'confidence': 0.89,
            'plate_number': None,  # Plate not detected
            'zone_id': 'ZONE-C3'
        },
        {
            'violation_type': ViolationType.BRT_LANE_VIOLATION,
            'vehicle_type': 'private_car',
            'confidence': 0.94,
            'plate_number': 'LAG-321-GH',
            'zone_id': 'ZONE-BRT-1'
        },
        {
            'violation_type': ViolationType.NO_PARKING_ZONE,
            'vehicle_type': 'keke_napep',
            'confidence': 0.88,
            'plate_number': 'EKY-654-IJ',
            'zone_id': 'ZONE-D4'
        },
        {
            'violation_type': ViolationType.LANE_VIOLATION,
            'vehicle_type': 'truck',
            'confidence': 0.91,
            'plate_number': 'ABJ-987-KL',
            'zone_id': 'ZONE-A1'
        },
        {
            'violation_type': ViolationType.NO_HELMET,
            'vehicle_type': 'okada',
            'confidence': 0.78,
            'plate_number': 'OGN-111-MN',
            'zone_id': 'ZONE-E5'
        },
    ]
    
    print("Creating sample violations...")
    
    for v in sample_violations:
        violation_id = db.create_violation(
            violation_type=v['violation_type'],
            vehicle_type=v['vehicle_type'],
            confidence=v['confidence'],
            plate_number=v['plate_number'],
            zone_id=v['zone_id'],
            camera_id='CAM-001',
            bbox=[100, 100, 300, 400],
            extra_data={'test': True}
        )
        print(f"  Created: {violation_id} ({v['violation_type'].value})")
    
    print(f"\nTotal sample violations created: {len(sample_violations)}")
    
    # Show stats
    stats = db.get_violation_stats()
    print(f"\nViolation Statistics:")
    print(f"  Total: {stats.get('total', 0)}")
    print(f"  Pending: {stats.get('pending', 0)}")
    print(f"  By Type: {stats.get('by_type', {})}")


if __name__ == "__main__":
    create_sample_violations()
