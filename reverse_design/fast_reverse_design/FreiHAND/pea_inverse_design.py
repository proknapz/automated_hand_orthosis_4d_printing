"""Phase 4B: Inverse Design Optimization using Pareto Evolutionary Algorithm (PEA).

This module integrates the Phase 4A design space with an evolutionary algorithm
to generate optimal orthosis designs by:
- Encoding designs as variable-length voxel configurations
- Defining multi-objective fitness functions (support, movement, efficiency, fabricability)
- Running PEA with design space constraints
- Ranking and selecting top candidate designs
- Exporting results for fabrication

Key Concepts:
1. Design Encoding: A design is a set of voxel indices where material is placed
2. Design Space Constraints: Mutations/crossover stay within anatomically valid regions
3. Multi-Objective Fitness:
   - f1: Support Quality (FE simulation or heuristic)
   - f2: Movement Preservation (DL model prediction or heuristic)
   - f3: Material Efficiency (minimize voxel count)
   - f4: Fabrication Feasibility (connectivity, printability)
4. Pareto Ranking: Non-dominated sorting to maintain diverse design front
5. Selection: Tournament selection among best designs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from FreiHAND.orthosis_design_space import DesignSpace, DesignZoneType, OrthosisDesignSpace


class DesignEncoding(Enum):
    """Encoding scheme for orthosis designs."""
    BINARY_GRID = 0      # Binary 3D grid (slow for sparse designs)
    VOXEL_SET = 1        # Set of voxel indices (fast for sparse designs)
    COMPRESSED = 2       # Compressed representation (future optimization)


@dataclass
class DesignIndividual:
    """A single orthosis design candidate in the population."""
    
    voxel_indices: NDArray[np.int32]  # Indices of voxels where material is placed
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    
    # Multi-objective fitness components
    support_quality: Optional[float] = None
    movement_preservation: Optional[float] = None
    material_efficiency: Optional[float] = None
    fabrication_feasibility: Optional[float] = None
    
    # Metadata
    generation: int = 0
    parent_ids: Tuple[int, int] = (-1, -1)
    
    def voxel_count(self) -> int:
        """Number of voxels in this design."""
        return len(self.voxel_indices)
    
    def voxel_volume_cm3(self, voxel_resolution_m: float = 0.002) -> float:
        """Volume in cm³."""
        voxel_volume_m3 = voxel_resolution_m ** 3
        voxel_volume_cm3 = voxel_volume_m3 * 1e6  # m³ to cm³
        return self.voxel_count() * voxel_volume_cm3
    
    def weight_grams(self, material_density_g_cm3: float = 1.2) -> float:
        """Estimated weight in grams."""
        return self.voxel_volume_cm3() * material_density_g_cm3
    
    def dominates(self, other: DesignIndividual) -> bool:
        """Check if this design dominates another in all objectives.
        
        For orthosis design:
        - Higher support_quality is better (≥)
        - Higher movement_preservation is better (≥)
        - Higher material_efficiency is better (≥)
        - Higher fabrication_feasibility is better (≥)
        
        Domination: self is better or equal in ALL and strictly better in at least ONE.
        """
        my_scores = [
            self.support_quality or 0.0,
            self.movement_preservation or 0.0,
            self.material_efficiency or 0.0,
            self.fabrication_feasibility or 0.0,
        ]
        other_scores = [
            other.support_quality or 0.0,
            other.movement_preservation or 0.0,
            other.material_efficiency or 0.0,
            other.fabrication_feasibility or 0.0,
        ]
        
        # Check: better or equal in all
        all_ge = all(m >= o for m, o in zip(my_scores, other_scores))
        # Check: strictly better in at least one
        any_g = any(m > o for m, o in zip(my_scores, other_scores))
        
        return all_ge and any_g


@dataclass
class PopulationStatistics:
    """Statistics about a PEA population."""
    
    generation: int
    population_size: int
    
    avg_voxel_count: float = 0.0
    min_voxel_count: int = 0
    max_voxel_count: int = 0
    
    avg_support: float = 0.0
    avg_movement: float = 0.0
    avg_efficiency: float = 0.0
    avg_fabricability: float = 0.0
    
    pareto_front_size: int = 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Generation {self.generation} Statistics:",
            f"  Population size: {self.population_size}",
            f"  Voxel count - avg: {self.avg_voxel_count:.0f}, min: {self.min_voxel_count}, max: {self.max_voxel_count}",
            f"  Avg fitness - Support: {self.avg_support:.3f}, Movement: {self.avg_movement:.3f}",
            f"              Efficiency: {self.avg_efficiency:.3f}, Fabricability: {self.avg_fabricability:.3f}",
            f"  Pareto front size: {self.pareto_front_size}",
        ]
        return "\n".join(lines)


class FitnessEvaluator:
    """Compute multi-objective fitness scores for orthosis designs."""
    
    def __init__(self, design_space: DesignSpace):
        """Initialize fitness evaluator with design space constraints.
        
        Args:
            design_space: Design space from Phase 4A with constraints
        """
        self.design_space = design_space
    
    def evaluate(self, individual: DesignIndividual) -> DesignIndividual:
        """Compute all fitness components for a design.
        
        Returns:
            Updated individual with fitness scores set
        """
        # f1: Support Quality (0-1, higher is better)
        individual.support_quality = self._evaluate_support_quality(individual)
        
        # f2: Movement Preservation (0-1, higher is better)
        individual.movement_preservation = self._evaluate_movement_preservation(individual)
        
        # f3: Material Efficiency (0-1, higher is better)
        individual.material_efficiency = self._evaluate_material_efficiency(individual)
        
        # f4: Fabrication Feasibility (0-1, higher is better)
        individual.fabrication_feasibility = self._evaluate_fabrication_feasibility(individual)
        
        return individual
    
    def _evaluate_support_quality(self, individual: DesignIndividual) -> float:
        """Evaluate how well the design supports critical regions.
        
        Heuristic: Average support priority of placed material voxels,
        weighted by region importance.
        
        Returns: Score 0-1 (1.0 = maximum support)
        """
        if len(individual.voxel_indices) == 0:
            return 0.0
        
        total_priority = 0.0
        for voxel_idx in individual.voxel_indices:
            # Convert linear index to tuple for lookup
            voxel_pos = self._index_to_pos(voxel_idx)
            support_priority = self.design_space.support_map.get(voxel_pos, 0.0)
            total_priority += support_priority
        
        avg_priority = total_priority / len(individual.voxel_indices)
        # Normalize to 0-1 range (max realistic priority ~0.95)
        return min(1.0, avg_priority / 0.95)
    
    def _evaluate_movement_preservation(self, individual: DesignIndividual) -> float:
        """Evaluate how well the design preserves hand movement capability.
        
        Heuristic: Average movement factor in placement zones.
        High movement factor (flexible material) = better movement preservation.
        
        Returns: Score 0-1 (1.0 = maximum flexibility)
        """
        if len(individual.voxel_indices) == 0:
            return 1.0  # Empty design is maximally flexible (but useless)
        
        total_movement = 0.0
        for voxel_idx in individual.voxel_indices:
            voxel_pos = self._index_to_pos(voxel_idx)
            movement_factor = self.design_space.movement_map.get(voxel_pos, 0.5)
            total_movement += movement_factor
        
        avg_movement = total_movement / len(individual.voxel_indices)
        return avg_movement  # Already 0-1
    
    def _evaluate_material_efficiency(self, individual: DesignIndividual) -> float:
        """Evaluate material efficiency (minimize weight while maintaining support).
        
        Heuristic: Ratio of material volume used relative to budget.
        Better score = using less material for same support quality.
        
        Returns: Score 0-1 (1.0 = most efficient, uses minimal material)
        """
        max_voxels = self.design_space.max_material_voxels
        voxel_count = len(individual.voxel_indices)
        
        if voxel_count == 0:
            return 1.0  # No material = most efficient (but non-functional)
        
        # Efficiency inversely proportional to material used
        # Full budget = 0.5, half budget = 0.75, quarter = 0.875, etc.
        utilization_fraction = voxel_count / max_voxels
        efficiency = 1.0 - (utilization_fraction * 0.5)
        return max(0.0, efficiency)
    
    def _evaluate_fabrication_feasibility(self, individual: DesignIndividual) -> float:
        """Evaluate feasibility of 3D printing this design.
        
        Heuristic factors:
        - Connectivity (no isolated voxels)
        - Avoids forbidden zones (keep-clear)
        - Reasonable wall thickness (not too thin/thick)
        
        Returns: Score 0-1 (1.0 = most printable)
        """
        if len(individual.voxel_indices) == 0:
            return 0.5  # Can't print empty design
        
        # Check: Are voxels in valid zones (not FORBIDDEN)?
        score = 0.0
        valid_voxels = 0
        
        for voxel_idx in individual.voxel_indices:
            voxel_pos = self._index_to_pos(voxel_idx)
            zone = self.design_space.zone_map.get(voxel_pos, DesignZoneType.SUPPORT)
            
            if zone != DesignZoneType.FORBIDDEN:
                valid_voxels += 1
        
        zone_score = valid_voxels / len(individual.voxel_indices)
        
        # Check: Design has some connectivity (very basic check)
        # For now, assume connected if voxel count is reasonable
        connectivity_score = 1.0 if len(individual.voxel_indices) >= 10 else 0.5
        
        # Average the scores
        score = (zone_score * 0.6) + (connectivity_score * 0.4)
        return score
    
    def _index_to_pos(self, linear_idx: int) -> Tuple[int, int, int]:
        """Convert linear voxel index to 3D grid position.
        
        Uses row-major (C-style) indexing:
        index = x * (Y * Z) + y * Z + z
        """
        x_size, y_size, z_size = self.design_space.grid_shape
        x = linear_idx // (y_size * z_size)
        remainder = linear_idx % (y_size * z_size)
        y = remainder // z_size
        z = remainder % z_size
        return (x, y, z)
    
    def _pos_to_index(self, pos: Tuple[int, int, int]) -> int:
        """Convert 3D position to linear voxel index."""
        x, y, z = pos
        y_size, z_size = self.design_space.grid_shape[1:]
        return x * (y_size * z_size) + y * z_size + z


class PEAOrthosisOptimizer:
    """Pareto Evolutionary Algorithm for orthosis design optimization."""
    
    def __init__(
        self,
        design_space: DesignSpace,
        population_size: int = 100,
        generations: int = 50,
        encoding: DesignEncoding = DesignEncoding.VOXEL_SET,
    ):
        """Initialize PEA optimizer.
        
        Args:
            design_space: Design space from Phase 4A
            population_size: Number of designs per generation
            generations: Number of generations to evolve
            encoding: How to represent designs
        """
        self.design_space = design_space
        self.population_size = population_size
        self.max_generations = generations
        self.encoding = encoding
        self.fitness_evaluator = FitnessEvaluator(design_space)
        
        self.population: List[DesignIndividual] = []
        self.pareto_front: List[DesignIndividual] = []
        self.generation_history: List[PopulationStatistics] = []
    
    def optimize(self) -> Tuple[List[DesignIndividual], List[PopulationStatistics]]:
        """Run PEA optimization loop.
        
        Returns:
            (final_pareto_front, generation_statistics)
        """
        print("[STEP 1] Initializing population...")
        self._initialize_population()
        
        for gen in range(self.max_generations):
            print(f"[GEN {gen+1}/{self.max_generations}] Evolution step...")
            
            # Evaluation
            print(f"  - Evaluating {len(self.population)} individuals...")
            self._evaluate_population()
            
            # Non-dominated sorting and Pareto front extraction
            print(f"  - Non-dominated sorting...")
            self._update_pareto_front()
            
            # Statistics
            stats = self._compute_statistics(gen)
            self.generation_history.append(stats)
            print(f"    {stats.summary()}")
            
            # Selection, crossover, mutation (if not last generation)
            if gen < self.max_generations - 1:
                print(f"  - Selection and reproduction...")
                self._selection_and_reproduction()
        
        print(f"[COMPLETE] Optimization finished with Pareto front size: {len(self.pareto_front)}")
        return self.pareto_front, self.generation_history
    
    def _initialize_population(self) -> None:
        """Create initial population from seed candidates + random designs."""
        self.population = []
        
        # Seed candidates from design space (80% of population)
        num_seeds = min(len(self.design_space.seed_candidates), 
                       int(self.population_size * 0.8))
        
        for i, seed_voxels in enumerate(self.design_space.seed_candidates[:num_seeds]):
            individual = DesignIndividual(
                voxel_indices=seed_voxels.copy(),
                generation=0,
            )
            self.population.append(individual)
            print(f"  [SEED {i+1}] {len(seed_voxels)} voxels")
        
        # Random designs (20% of population)
        num_random = self.population_size - len(self.population)
        for i in range(num_random):
            random_voxels = self._generate_random_design()
            individual = DesignIndividual(
                voxel_indices=random_voxels,
                generation=0,
            )
            self.population.append(individual)
        
        print(f"  [OK] Initial population: {len(self.population)} individuals")
    
    def _generate_random_design(self) -> NDArray[np.int32]:
        """Generate a random orthosis design respecting design space constraints."""
        # Randomly select 30-60% of available support voxels
        available_voxels = list(self.design_space.support_map.keys())
        
        if not available_voxels:
            return np.array([], dtype=np.int32)
        
        fraction = np.random.uniform(0.3, 0.6)
        num_voxels = int(len(available_voxels) * fraction)
        num_voxels = min(num_voxels, self.design_space.max_material_voxels)
        
        selected = np.random.choice(len(available_voxels), size=num_voxels, replace=False)
        selected_voxels = [available_voxels[i] for i in selected]
        
        # Convert positions to linear indices
        indices = np.array([self.fitness_evaluator._pos_to_index(pos) 
                           for pos in selected_voxels], dtype=np.int32)
        return np.sort(indices)
    
    def _evaluate_population(self) -> None:
        """Evaluate all individuals in population."""
        for individual in self.population:
            if individual.support_quality is None:
                self.fitness_evaluator.evaluate(individual)
    
    def _update_pareto_front(self) -> None:
        """Extract non-dominated solutions (Pareto front) from population."""
        # Non-dominated sorting: O(n²) for small populations
        front = []
        
        for candidate in self.population:
            dominated = False
            
            for other in self.population:
                if other is candidate:
                    continue
                if other.dominates(candidate):
                    dominated = True
                    break
            
            if not dominated:
                front.append(candidate)
        
        self.pareto_front = front
    
    def _compute_statistics(self, generation: int) -> PopulationStatistics:
        """Compute population statistics for reporting."""
        voxel_counts = [ind.voxel_count() for ind in self.population]
        support_scores = [ind.support_quality or 0.0 for ind in self.population]
        movement_scores = [ind.movement_preservation or 0.0 for ind in self.population]
        efficiency_scores = [ind.material_efficiency or 0.0 for ind in self.population]
        fab_scores = [ind.fabrication_feasibility or 0.0 for ind in self.population]
        
        return PopulationStatistics(
            generation=generation,
            population_size=len(self.population),
            avg_voxel_count=np.mean(voxel_counts),
            min_voxel_count=int(np.min(voxel_counts)),
            max_voxel_count=int(np.max(voxel_counts)),
            avg_support=np.mean(support_scores),
            avg_movement=np.mean(movement_scores),
            avg_efficiency=np.mean(efficiency_scores),
            avg_fabricability=np.mean(fab_scores),
            pareto_front_size=len(self.pareto_front),
        )
    
    def _selection_and_reproduction(self) -> None:
        """Tournament selection, crossover, and mutation to create next generation."""
        new_population = []
        
        # Elitism: Keep Pareto front
        new_population.extend(self.pareto_front)
        
        # Fill rest of population via tournament selection + variation
        while len(new_population) < self.population_size:
            # Tournament selection (2 individuals, pick better one)
            tournament_size = 2
            tournament = np.random.choice(len(self.population), tournament_size, replace=False)
            parent1 = self.population[tournament[0]]
            parent2 = self.population[tournament[1]]
            
            # Choose operation
            op = np.random.choice(['crossover', 'mutation', 'copy'], p=[0.4, 0.4, 0.2])
            
            if op == 'crossover':
                child = self._crossover(parent1, parent2)
            elif op == 'mutation':
                child = self._mutate(parent1)
            else:  # copy
                child = DesignIndividual(
                    voxel_indices=parent1.voxel_indices.copy(),
                    generation=parent1.generation + 1,
                    parent_ids=(tournament[0], tournament[1]),
                )
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def _crossover(self, parent1: DesignIndividual, parent2: DesignIndividual) -> DesignIndividual:
        """Crossover two parents to create child.
        
        Uniform crossover: Each voxel position has 50% chance from each parent.
        """
        # Combine voxel sets
        voxels_p1 = set(parent1.voxel_indices)
        voxels_p2 = set(parent2.voxel_indices)
        
        # Uniform crossover on union
        all_voxels = voxels_p1.union(voxels_p2)
        child_voxels = set()
        
        for voxel in all_voxels:
            in_p1 = voxel in voxels_p1
            in_p2 = voxel in voxels_p2
            
            if in_p1 and in_p2:
                # Both parents: 50% chance
                if np.random.rand() < 0.5:
                    child_voxels.add(voxel)
            elif in_p1:
                # Only p1: 50% chance
                if np.random.rand() < 0.5:
                    child_voxels.add(voxel)
            elif in_p2:
                # Only p2: 50% chance
                if np.random.rand() < 0.5:
                    child_voxels.add(voxel)
        
        # Ensure doesn't exceed budget
        if len(child_voxels) > self.design_space.max_material_voxels:
            child_voxels = set(np.random.choice(
                list(child_voxels),
                size=self.design_space.max_material_voxels,
                replace=False
            ))
        
        child = DesignIndividual(
            voxel_indices=np.array(sorted(child_voxels), dtype=np.int32),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=(id(parent1), id(parent2)),
        )
        return child
    
    def _mutate(self, parent: DesignIndividual) -> DesignIndividual:
        """Mutate a design via add/remove voxels or local perturbation.
        
        Mutation types (equal probability):
        1. Add 10-20% random voxels from design space
        2. Remove 10-20% random voxels
        3. Local perturbation: Add/remove around existing material
        """
        child_voxels = set(parent.voxel_indices)
        mutation_type = np.random.choice(['add', 'remove', 'local'])
        
        if mutation_type == 'add':
            # Add random voxels
            available = self.design_space.max_material_voxels - len(child_voxels)
            if available > 0:
                num_add = max(1, int(len(child_voxels) * np.random.uniform(0.1, 0.2)))
                num_add = min(num_add, available)
                
                # Try to add from support regions
                support_voxels = list(self.design_space.support_map.keys())
                support_indices = np.array([self.fitness_evaluator._pos_to_index(pos)
                                           for pos in support_voxels], dtype=np.int32)
                available_to_add = set(support_indices) - child_voxels
                
                if available_to_add:
                    new_voxels = np.random.choice(
                        list(available_to_add),
                        size=min(num_add, len(available_to_add)),
                        replace=False
                    )
                    child_voxels.update(new_voxels)
        
        elif mutation_type == 'remove':
            # Remove random voxels
            num_remove = max(1, int(len(child_voxels) * np.random.uniform(0.1, 0.2)))
            if len(child_voxels) > 0:
                to_remove = np.random.choice(
                    list(child_voxels),
                    size=min(num_remove, len(child_voxels)),
                    replace=False
                )
                for v in to_remove:
                    child_voxels.discard(v)
        
        else:  # local perturbation
            # Add/remove around existing material
            if child_voxels:
                num_local = max(1, int(len(child_voxels) * np.random.uniform(0.05, 0.15)))
                for _ in range(num_local):
                    op = np.random.choice(['add', 'remove'])
                    if op == 'add' and len(child_voxels) < self.design_space.max_material_voxels:
                        # Pick random existing voxel and add neighbor
                        existing = np.random.choice(list(child_voxels))
                        # Try to add nearby voxel (±1 in some dimension)
                        new_voxel = existing + np.random.choice([-1, 0, 1], size=1)[0]
                        if 0 <= new_voxel < np.prod(self.design_space.grid_shape):
                            child_voxels.add(new_voxel)
                    elif op == 'remove' and child_voxels:
                        to_remove = np.random.choice(list(child_voxels))
                        child_voxels.discard(to_remove)
        
        child = DesignIndividual(
            voxel_indices=np.array(sorted(child_voxels), dtype=np.int32),
            generation=parent.generation + 1,
            parent_ids=(id(parent), -1),
        )
        return child
    
    def export_results(self, output_dir: Path) -> None:
        """Export final Pareto front and statistics to files.
        
        Creates:
        - designs.json: All Pareto front designs with scores
        - statistics.json: Generation-wise statistics
        - top_design.json: Best design (by support quality)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export all Pareto front designs
        designs_data = []
        for i, design in enumerate(self.pareto_front):
            designs_data.append({
                "rank": i + 1,
                "voxel_count": int(design.voxel_count()),
                "weight_grams": float(design.weight_grams()),
                "support_quality": float(design.support_quality or 0.0),
                "movement_preservation": float(design.movement_preservation or 0.0),
                "material_efficiency": float(design.material_efficiency or 0.0),
                "fabrication_feasibility": float(design.fabrication_feasibility or 0.0),
                "voxel_indices": [int(i) for i in design.voxel_indices[:100]]  # First 100 for preview
            })
        
        with open(output_dir / "designs.json", "w") as f:
            json.dump(designs_data, f, indent=2)
        print(f"[OK] Exported {len(designs_data)} Pareto front designs")
        
        # Export statistics
        stats_data = [
            {
                "generation": s.generation,
                "population_size": s.population_size,
                "avg_voxel_count": float(s.avg_voxel_count),
                "avg_support": float(s.avg_support),
                "avg_movement": float(s.avg_movement),
                "avg_efficiency": float(s.avg_efficiency),
                "avg_fabricability": float(s.avg_fabricability),
                "pareto_front_size": s.pareto_front_size,
            }
            for s in self.generation_history
        ]
        
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats_data, f, indent=2)
        print(f"[OK] Exported {len(stats_data)} generation statistics")
        
        # Export best design
        if self.pareto_front:
            best = max(self.pareto_front, key=lambda d: d.support_quality or 0.0)
            with open(output_dir / "top_design.json", "w") as f:
                json.dump({
                    "description": "Best design by support quality",
                    "voxel_count": int(best.voxel_count()),
                    "weight_grams": float(best.weight_grams()),
                    "support_quality": float(best.support_quality or 0.0),
                    "movement_preservation": float(best.movement_preservation or 0.0),
                    "material_efficiency": float(best.material_efficiency or 0.0),
                    "fabrication_feasibility": float(best.fabrication_feasibility or 0.0),
                }, f, indent=2)
            print(f"[OK] Exported top design (support: {best.support_quality:.3f})")


if __name__ == "__main__":
    print("="*70)
    print("PHASE 4B: INVERSE DESIGN OPTIMIZATION")
    print("="*70)
    print()
    print("[INFO] This module requires Phase 4A design space to be initialized.")
    print("[INFO] Use load_and_build_design_space() then PEAOrthosisOptimizer")
    print()
