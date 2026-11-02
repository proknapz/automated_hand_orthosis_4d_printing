"""Orthosis design space definition and constraint generation for Phase 4A.

This module bridges Phase 2-3 outputs (segmentation, voxelization) to Phase 4B
inverse design by:
- Loading hand segmentation and voxelization data
- Mapping anatomical priorities to voxel-level support maps
- Defining movement restriction zones
- Generating design candidate seeds
- Providing visualization tools for design space validation

The design space constrains orthosis material placement to:
1. Anatomically valid regions (surface voxels only)
2. Clinically appropriate support levels (from design_parameters.json)
3. Movement-preserving zones (flexible for joint motion)
4. Material budget constraints (weight and voxel count limits)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from FreiHAND.hand_segmentation import AnatomicalRegion, HandSegmentation, SurfaceType
from FreiHAND.surface_voxelization import HandVoxelization, VoxelData


class DesignZoneType(Enum):
    """Classification of design space zones by their design role."""
    
    FORBIDDEN = 0      # Cannot place material (keep-clear, joint bends)
    FLEXIBLE = 1       # Can place material but should remain thin/flexible
    SUPPORT = 2        # Primary support region, can be full thickness
    REINFORCE = 3      # Reinforced support region, should be stiff


@dataclass
class DesignConstraint:
    """A constraint on orthosis design."""
    
    zone_type: DesignZoneType
    region: AnatomicalRegion
    support_priority: float  # 0.0-1.0, from design_parameters.json
    movement_restriction: float  # 0.0-1.0, how much movement must be preserved
    voxel_indices: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    thickness_mm: Tuple[float, float] = (0.0, 8.0)  # Min/max thickness for this zone
    fill_fraction: Tuple[float, float] = (0.0, 1.0)  # Min/max material fill for this zone


@dataclass
class DesignSpace:
    """Complete orthosis design space with all constraints and metadata."""
    
    # Grid and voxel information
    grid_shape: Tuple[int, int, int]
    grid_origin: NDArray[np.float64]
    voxel_resolution: float  # meters
    
    # Constraint maps (same shape as grid, stored as dict for sparsity)
    zone_map: Dict[Tuple[int, int, int], DesignZoneType]  # Voxel -> zone type
    support_map: Dict[Tuple[int, int, int], float]  # Voxel -> support priority (0-1)
    movement_map: Dict[Tuple[int, int, int], float]  # Voxel -> movement preservation (0-1)
    thickness_map: Dict[Tuple[int, int, int], Tuple[float, float]]  # Voxel -> thickness range
    
    # Design constraints
    constraints: List[DesignConstraint] = field(default_factory=list)
    max_material_voxels: int = 10000
    material_density: float = 1.2  # g/cm^3
    
    # Design parameters from analyzer
    design_params: Dict = field(default_factory=dict)
    
    # Seed candidates (initial designs for PEA)
    seed_candidates: List[NDArray[np.int32]] = field(default_factory=list)
    
    def total_voxels(self) -> int:
        """Total voxels in grid."""
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
    
    def available_voxels(self) -> int:
        """Count of voxels where material can be placed."""
        return len(self.support_map)
    
    def summary(self) -> str:
        """Return human-readable summary of design space."""
        lines = [
            "Design Space Summary:",
            f"  Grid shape: {self.grid_shape}",
            f"  Total voxels: {self.total_voxels():,}",
            f"  Available for material: {self.available_voxels():,}",
            f"  Max material voxels: {self.max_material_voxels:,}",
            f"  Material density: {self.material_density} g/cm^3",
            f"  Voxel resolution: {self.voxel_resolution*1000:.1f} mm",
            f"  Constraints: {len(self.constraints)}",
            f"  Seed candidates: {len(self.seed_candidates)}",
        ]
        return "\n".join(lines)


class OrthosisDesignSpace:
    """Build and manage orthosis design space from Phase 2-3 data."""
    
    def __init__(self, voxel_data: VoxelData, segmentation: HandSegmentation,
                 design_params_path: Optional[Path] = None):
        """Initialize design space builder.
        
        Args:
            voxel_data: Voxelized hand from Phase 3
            segmentation: Hand segmentation from Phase 2
            design_params_path: Path to design_parameters.json from analyzer
        """
        self.voxel_data = voxel_data
        self.segmentation = segmentation
        self.design_params = self._load_design_params(design_params_path)
        
        self.design_space = None
    
    def _load_design_params(self, params_path: Optional[Path]) -> Dict:
        """Load design parameters from JSON file."""
        if params_path is None:
            # Default to FreiHAND directory
            params_path = Path(__file__).parent / "design_parameters.json"
        
        if params_path.exists():
            with open(params_path) as f:
                return json.load(f)
        else:
            # Fallback defaults from analyzer
            return {
                "weight_budget_grams": 80,
                "max_material_voxels": 10000,
                "material_density_g_cm3": 1.2,
                "support_priorities": {
                    "wrist": 0.95,
                    "thumb": 0.85,
                    "index": 0.8,
                    "middle": 0.8,
                    "ring": 0.75,
                    "pinky": 0.75,
                    "palm": 0.7,
                },
                "movement_restrictions": {
                    "wrist": 0.3,
                    "thumb": 0.5,
                    "index": 0.6,
                    "middle": 0.6,
                    "ring": 0.6,
                    "pinky": 0.5,
                    "palm": 0.2,
                }
            }
    
    def build(self) -> DesignSpace:
        """Build complete design space with all constraints."""
        print("[STEP 1] Identifying support regions...")
        zone_map, support_map = self._identify_support_regions()
        
        print("[STEP 2] Defining movement zones...")
        movement_map = self._define_movement_zones(zone_map)
        
        print("[STEP 3] Computing thickness maps...")
        thickness_map = self._create_thickness_maps(zone_map, movement_map)
        
        print("[STEP 4] Generating design constraints...")
        constraints = self._generate_design_constraints(zone_map, support_map, movement_map)
        
        print("[STEP 5] Creating design space...")
        self.design_space = DesignSpace(
            grid_shape=self.voxel_data.shape,
            grid_origin=self.voxel_data.origin,
            voxel_resolution=self.voxel_data.resolution,
            zone_map=zone_map,
            support_map=support_map,
            movement_map=movement_map,
            thickness_map=thickness_map,
            constraints=constraints,
            max_material_voxels=self.design_params.get("max_material_voxels", 10000),
            material_density=self.design_params.get("material_density_g_cm3", 1.2),
            design_params=self.design_params,
        )
        
        print("[STEP 6] Generating seed candidates...")
        self.design_space.seed_candidates = self._generate_seed_candidates()
        
        return self.design_space
    
    def _identify_support_regions(self) -> Tuple[Dict, Dict]:
        """Map anatomical regions to support priority levels.
        
        Returns:
            (zone_map, support_map): Voxel-indexed maps of design zones and priorities
        """
        zone_map = {}
        support_map = {}
        
        # Map voxels by their region and surface type
        support_priorities = self.design_params["support_priorities"]
        
        for i, voxel_pos in enumerate(self.voxel_data.voxel_positions):
            pos_tuple = tuple(voxel_pos)
            
            # Get region for this voxel
            region_id = self.voxel_data.voxel_region_ids[i]
            region_name = self.voxel_data.region_id_to_name.get(region_id, "unknown")
            
            try:
                region = AnatomicalRegion(region_name)
            except ValueError:
                region = AnatomicalRegion.UNKNOWN
            
            # Get surface type and weight
            surface_weight = self.voxel_data.voxel_weights[i]
            
            # Determine zone type based on surface weight
            if surface_weight <= 0.0:
                zone_type = DesignZoneType.FORBIDDEN
            elif surface_weight < 0.3:
                zone_type = DesignZoneType.FLEXIBLE
            elif surface_weight < 0.7:
                zone_type = DesignZoneType.SUPPORT
            else:
                zone_type = DesignZoneType.REINFORCE
            
            zone_map[pos_tuple] = zone_type
            
            # Get support priority for this region
            priority = support_priorities.get(region.value, 0.7)
            # Scale by surface weight
            support_map[pos_tuple] = priority * max(0.0, surface_weight)
        
        print(f"[OK] Mapped {len(zone_map)} voxels to support regions")
        return zone_map, support_map
    
    def _define_movement_zones(self, zone_map: Dict) -> Dict:
        """Define movement restriction zones for joint flexibility.
        
        Returns:
            movement_map: Voxel -> movement preservation factor (1.0 = fully flexible)
        """
        movement_map = {}
        movement_restrictions = self.design_params["movement_restrictions"]
        
        for pos_tuple, zone_type in zone_map.items():
            # Determine region (try to infer from neighboring voxels or use default)
            # For now, apply movement restriction uniformly by zone type
            
            if zone_type == DesignZoneType.FORBIDDEN:
                movement_factor = 0.0  # No material here anyway
            elif zone_type == DesignZoneType.FLEXIBLE:
                movement_factor = 0.8  # Mostly flexible
            elif zone_type == DesignZoneType.SUPPORT:
                movement_factor = 0.5  # Moderate stiffness
            else:  # REINFORCE
                movement_factor = 0.2  # Stiff
            
            movement_map[pos_tuple] = movement_factor
        
        print(f"[OK] Defined movement restrictions for {len(movement_map)} voxels")
        return movement_map
    
    def _create_thickness_maps(self, zone_map: Dict, movement_map: Dict) -> Dict:
        """Create thickness constraints for each voxel.
        
        Returns:
            thickness_map: Voxel -> (min_mm, max_mm) thickness range
        """
        thickness_map = {}
        
        for pos_tuple, zone_type in zone_map.items():
            movement_factor = movement_map.get(pos_tuple, 0.5)
            
            if zone_type == DesignZoneType.FORBIDDEN:
                min_thick, max_thick = 0.0, 0.0
            elif zone_type == DesignZoneType.FLEXIBLE:
                # Thin for flexibility
                min_thick, max_thick = 0.0, 2.0
            elif zone_type == DesignZoneType.SUPPORT:
                # Moderate thickness
                min_thick, max_thick = 2.0, 5.0
            else:  # REINFORCE
                # Full thickness for stiffness
                min_thick, max_thick = 4.0, 8.0
            
            # Adjust based on movement factor (higher = more flexible = thinner)
            adjusted_max = max_thick * (1.0 - movement_factor * 0.5)
            thickness_map[pos_tuple] = (min_thick, adjusted_max)
        
        print(f"[OK] Created thickness maps for {len(thickness_map)} voxels")
        return thickness_map
    
    def _generate_design_constraints(self, zone_map: Dict, support_map: Dict,
                                     movement_map: Dict) -> List[DesignConstraint]:
        """Generate design constraints from maps.
        
        Returns:
            List of DesignConstraint objects for each anatomical region
        """
        constraints = []
        support_priorities = self.design_params["support_priorities"]
        movement_restrictions = self.design_params["movement_restrictions"]
        
        # Create constraint per anatomical region
        for region_name, priority in support_priorities.items():
            try:
                region = AnatomicalRegion(region_name)
            except ValueError:
                continue
            
            # Collect voxels for this region
            region_voxels = []
            avg_support = 0.0
            count = 0
            
            for pos_tuple, support in support_map.items():
                # Try to match by region (would need region data per voxel)
                # For now, collect by support level near priority
                if abs(support - priority) < 0.15:
                    region_voxels.append(pos_tuple)
                    avg_support += support
                    count += 1
            
            if region_voxels:
                avg_support = avg_support / count if count > 0 else priority
                movement_rest = movement_restrictions.get(region_name, 0.5)
                
                zone_type = zone_map.get(region_voxels[0], DesignZoneType.SUPPORT)
                thickness_range = self._thickness_for_zone(zone_type, movement_rest)
                
                constraint = DesignConstraint(
                    zone_type=zone_type,
                    region=region,
                    support_priority=avg_support,
                    movement_restriction=movement_rest,
                    voxel_indices=np.array(region_voxels, dtype=np.int32),
                    thickness_mm=thickness_range,
                    fill_fraction=self._fill_fraction_for_priority(priority),
                )
                constraints.append(constraint)
        
        print(f"[OK] Generated {len(constraints)} design constraints")
        return constraints
    
    def _thickness_for_zone(self, zone_type: DesignZoneType,
                           movement_rest: float) -> Tuple[float, float]:
        """Determine thickness range for a zone."""
        base = {
            DesignZoneType.FORBIDDEN: (0.0, 0.0),
            DesignZoneType.FLEXIBLE: (0.0, 2.0),
            DesignZoneType.SUPPORT: (2.0, 5.0),
            DesignZoneType.REINFORCE: (4.0, 8.0),
        }
        min_t, max_t = base.get(zone_type, (0.0, 8.0))
        
        # Reduce max thickness for high movement requirement (low movement_rest)
        if movement_rest > 0.6:
            max_t *= 0.7
        
        return (min_t, max_t)
    
    def _fill_fraction_for_priority(self, priority: float) -> Tuple[float, float]:
        """Determine material fill fraction range for a support priority."""
        if priority >= 0.9:
            return (0.6, 1.0)  # High priority: dense fill
        elif priority >= 0.75:
            return (0.4, 0.8)  # Medium priority: moderate fill
        else:
            return (0.2, 0.6)  # Low priority: sparse fill
    
    def _generate_seed_candidates(self, num_seeds: int = 3) -> List[NDArray[np.int32]]:
        """Generate initial design candidates to seed PEA.
        
        Strategies:
        1. Maximum support (all high-priority regions)
        2. Balanced (support + movement balance)
        3. Minimal (lowest material usage while maintaining support)
        
        Returns:
            List of candidate orthosis designs (voxel index arrays)
        """
        seeds = []
        support_map = self.design_space.support_map
        zone_map = self.design_space.zone_map
        movement_map = self.design_space.movement_map
        
        if not support_map:
            print("[WARN] No support map available, skipping seed generation")
            return seeds
        
        # Convert maps to sortable lists
        voxel_priorities = [(pos, pri) for pos, pri in support_map.items()]
        voxel_priorities.sort(key=lambda x: x[1], reverse=True)
        
        max_voxels = self.design_space.max_material_voxels
        
        # Seed 1: Maximum support (add until budget)
        seed1 = []
        for pos, priority in voxel_priorities:
            if len(seed1) >= max_voxels * 0.9:  # Use 90% of budget
                break
            zone = zone_map.get(pos, DesignZoneType.SUPPORT)
            if zone != DesignZoneType.FORBIDDEN:
                seed1.append(pos)
        seeds.append(np.array(seed1, dtype=np.int32))
        
        # Seed 2: Balanced (weight by both support and movement)
        voxel_balanced = []
        for pos, priority in voxel_priorities:
            movement = movement_map.get(pos, 0.5)
            # Balance: higher support AND lower movement restriction (stiffer) = good
            score = priority * (1.0 - movement)
            zone = zone_map.get(pos, DesignZoneType.SUPPORT)
            if zone != DesignZoneType.FORBIDDEN:
                voxel_balanced.append((pos, score))
        
        voxel_balanced.sort(key=lambda x: x[1], reverse=True)
        seed2 = [pos for pos, _ in voxel_balanced[:int(max_voxels * 0.6)]]
        seeds.append(np.array(seed2, dtype=np.int32))
        
        # Seed 3: Minimal (only highest priority regions)
        seed3 = []
        for pos, priority in voxel_priorities:
            if len(seed3) >= max_voxels * 0.3:  # Use 30% of budget
                break
            if priority >= 0.8:  # Only wrist and high-priority fingers
                zone = zone_map.get(pos, DesignZoneType.SUPPORT)
                if zone != DesignZoneType.FORBIDDEN:
                    seed3.append(pos)
        seeds.append(np.array(seed3, dtype=np.int32))
        
        print(f"[OK] Generated {len(seeds)} seed candidates")
        print(f"  Seed 1 (maximum support): {len(seed1)} voxels ({100*len(seed1)/max_voxels:.1f}% budget)")
        print(f"  Seed 2 (balanced): {len(seed2)} voxels ({100*len(seed2)/max_voxels:.1f}% budget)")
        print(f"  Seed 3 (minimal): {len(seed3)} voxels ({100*len(seed3)/max_voxels:.1f}% budget)")
        
        return seeds
    
    def visualize_design_space(self, output_path: Optional[Path] = None) -> None:
        """Visualize design space in 3D (requires matplotlib).
        
        Args:
            output_path: Path to save visualization PNG
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("[WARN] Matplotlib not available, skipping visualization")
            return
        
        if self.design_space is None:
            print("[WARN] Design space not built, call build() first")
            return
        
        # Separate voxels by zone type
        zones_by_type = {zone_type: [] for zone_type in DesignZoneType}
        for pos, zone_type in self.design_space.zone_map.items():
            zones_by_type[zone_type].append(pos)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {
            DesignZoneType.FORBIDDEN: 'red',
            DesignZoneType.FLEXIBLE: 'yellow',
            DesignZoneType.SUPPORT: 'cyan',
            DesignZoneType.REINFORCE: 'green',
        }
        
        labels_added = set()
        
        for zone_type, positions in zones_by_type.items():
            if not positions:
                continue
            
            positions = np.array(positions)
            color = colors.get(zone_type, 'gray')
            label = zone_type.name if zone_type not in labels_added else None
            
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=color, s=5, alpha=0.3, label=label)
            
            if label:
                labels_added.add(zone_type)
        
        # Plot seed candidates
        for i, seed in enumerate(self.design_space.seed_candidates):
            if len(seed) > 0:
                ax.scatter(seed[:, 0], seed[:, 1], seed[:, 2],
                          c='blue', s=20, alpha=0.6,
                          marker='s', label=f'Seed {i+1}' if i == 0 else None)
        
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        ax.set_zlabel('Z (voxels)')
        ax.set_title('Orthosis Design Space')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=100)
            print(f"[OK] Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


def load_and_build_design_space(
    hand_data_path: Path,
    segmentation_path: Optional[Path] = None,
    design_params_path: Optional[Path] = None,
) -> DesignSpace:
    """Convenience function to load Phase 2-3 data and build design space.
    
    Args:
        hand_data_path: Path to voxelized hand data (from Phase 3)
        segmentation_path: Path to hand segmentation data (from Phase 2)
        design_params_path: Path to design_parameters.json
    
    Returns:
        Complete DesignSpace ready for Phase 4B optimization
    """
    print("Loading Phase 3 voxelization data...")
    voxel_loader = HandVoxelization()
    voxel_data = voxel_loader.load_from_file(hand_data_path)
    
    print("Loading Phase 2 segmentation data...")
    segmentation = HandSegmentation()
    if segmentation_path:
        # TODO: Implement load_from_file for segmentation
        pass
    
    print("Building design space...")
    builder = OrthosisDesignSpace(voxel_data, segmentation, design_params_path)
    design_space = builder.build()
    
    print(design_space.summary())
    
    return design_space


if __name__ == "__main__":
    print("="*70)
    print("PHASE 4A: ORTHOSIS DESIGN SPACE")
    print("="*70)
    print()
    
    print("[INFO] This module requires Phase 2-3 data to be loaded and instantiated.")
    print("[INFO] Use load_and_build_design_space() or OrthosisDesignSpace class directly.")
    print()
    print("[TODO] Implement mock test with synthetic data")
