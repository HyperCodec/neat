use neat::*;
use std::f32::consts::PI;

// ==========================================================================
// SIMULATION CONSTANTS - Adjust these to experiment with different dynamics
// ==========================================================================

// World/Environment Settings
const WORLD_WIDTH: f32 = 800.0;
const WORLD_HEIGHT: f32 = 600.0;
const INITIAL_FOOD_COUNT: usize = 20;
const FOOD_RESPAWN_THRESHOLD: usize = 10;
const FOOD_DETECTION_DISTANCE: f32 = 10.0;

// Energy/Food Settings
const BASE_FOOD_ENERGY: f32 = 20.0; // Energy from each food item
const STRENGTH_ENERGY_MULTIPLIER: f32 = 10.0; // Extra energy per strength stat
const MOVEMENT_ENERGY_COST: f32 = 0.2; // Cost per unit of movement
const IDLE_ENERGY_COST: f32 = 0.1; // Cost per timestep just existing

// Fitness Settings
const FITNESS_PER_FOOD: f32 = 100.0; // Points per food eaten

// Physical Stats - Min/Max Bounds
const SPEED_MIN: f32 = 0.5;
const SPEED_MAX: f32 = 6.0;
const STRENGTH_MIN: f32 = 0.2;
const STRENGTH_MAX: f32 = 4.0;
const SENSE_RANGE_MIN: f32 = 30.0;
const SENSE_RANGE_MAX: f32 = 250.0;
const ENERGY_CAPACITY_MIN: f32 = 50.0;
const ENERGY_CAPACITY_MAX: f32 = 400.0;

// Physical Stats - Initial Generation Range
const SPEED_INIT_MIN: f32 = 1.0;
const SPEED_INIT_MAX: f32 = 5.0;
const STRENGTH_INIT_MIN: f32 = 0.5;
const STRENGTH_INIT_MAX: f32 = 3.0;
const SENSE_RANGE_INIT_MIN: f32 = 50.0;
const SENSE_RANGE_INIT_MAX: f32 = 200.0;
const ENERGY_CAPACITY_INIT_MIN: f32 = 100.0;
const ENERGY_CAPACITY_INIT_MAX: f32 = 300.0;

// Mutation Settings
const SPEED_MUTATION_PROB: f32 = 0.3;
const SPEED_MUTATION_RANGE: f32 = 0.5;
const STRENGTH_MUTATION_PROB: f32 = 0.2;
const STRENGTH_MUTATION_RANGE: f32 = 0.3;
const SENSE_MUTATION_PROB: f32 = 0.2;
const SENSE_MUTATION_RANGE: f32 = 20.0;
const CAPACITY_MUTATION_PROB: f32 = 0.2;
const CAPACITY_MUTATION_RANGE: f32 = 30.0;

// Genetic Algorithm Settings
const POPULATION_SIZE: usize = 150;
const HIGHEST_GENERATION: usize = 250;
const SIMULATION_TIMESTEPS: usize = 500;
const MUTATION_RATE: f32 = 0.3;

/// Mutation settings for physical stats
#[derive(Clone, Debug)]
struct PhysicalStatsMutationSettings {
    speed_prob: f32,
    speed_range: f32,
    strength_prob: f32,
    strength_range: f32,
    sense_prob: f32,
    sense_range: f32,
    capacity_prob: f32,
    capacity_range: f32,
}

impl Default for PhysicalStatsMutationSettings {
    fn default() -> Self {
        Self {
            speed_prob: SPEED_MUTATION_PROB,
            speed_range: SPEED_MUTATION_RANGE,
            strength_prob: STRENGTH_MUTATION_PROB,
            strength_range: STRENGTH_MUTATION_RANGE,
            sense_prob: SENSE_MUTATION_PROB,
            sense_range: SENSE_MUTATION_RANGE,
            capacity_prob: CAPACITY_MUTATION_PROB,
            capacity_range: CAPACITY_MUTATION_RANGE,
        }
    }
}

/// Physical traits/stats for an organism
#[derive(Clone, Debug, PartialEq)]
struct PhysicalStats {
    /// Speed multiplier (faster = longer strides but more energy cost)
    speed: f32,
    /// Strength stat (affects energy from food)
    strength: f32,
    /// Sense range (how far it can detect food)
    sense_range: f32,
    /// Energy capacity (larger = can go longer without food)
    energy_capacity: f32,
}

impl PhysicalStats {
    fn clamp(&mut self) {
        self.speed = self.speed.clamp(SPEED_MIN, SPEED_MAX);
        self.strength = self.strength.clamp(STRENGTH_MIN, STRENGTH_MAX);
        self.sense_range = self.sense_range.clamp(SENSE_RANGE_MIN, SENSE_RANGE_MAX);
        self.energy_capacity = self
            .energy_capacity
            .clamp(ENERGY_CAPACITY_MIN, ENERGY_CAPACITY_MAX);
    }
}

impl GenerateRandom for PhysicalStats {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        let mut stats = PhysicalStats {
            speed: rng.random_range(SPEED_INIT_MIN..SPEED_INIT_MAX),
            strength: rng.random_range(STRENGTH_INIT_MIN..STRENGTH_INIT_MAX),
            sense_range: rng.random_range(SENSE_RANGE_INIT_MIN..SENSE_RANGE_INIT_MAX),
            energy_capacity: rng.random_range(ENERGY_CAPACITY_INIT_MIN..ENERGY_CAPACITY_INIT_MAX),
        };
        stats.clamp();
        stats
    }
}

impl RandomlyMutable for PhysicalStats {
    type Context = PhysicalStatsMutationSettings;

    fn mutate(&mut self, context: &Self::Context, _severity: f32, rng: &mut impl rand::Rng) {
        if rng.random::<f32>() < context.speed_prob {
            self.speed += rng.random_range(-context.speed_range..context.speed_range);
        }
        if rng.random::<f32>() < context.strength_prob {
            self.strength += rng.random_range(-context.strength_range..context.strength_range);
        }
        if rng.random::<f32>() < context.sense_prob {
            self.sense_range += rng.random_range(-context.sense_range..context.sense_range);
        }
        if rng.random::<f32>() < context.capacity_prob {
            self.energy_capacity +=
                rng.random_range(-context.capacity_range..context.capacity_range);
        }
        self.clamp();
    }
}

impl Crossover for PhysicalStats {
    type Context = PhysicalStatsMutationSettings;

    fn crossover(
        &self,
        other: &Self,
        context: &Self::Context,
        _severity: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let mut child = PhysicalStats {
            speed: (self.speed + other.speed) / 2.0
                + rng.random_range(-context.speed_range..context.speed_range),
            strength: (self.strength + other.strength) / 2.0
                + rng.random_range(-context.strength_range..context.strength_range),
            sense_range: (self.sense_range + other.sense_range) / 2.0
                + rng.random_range(-context.sense_range..context.sense_range),
            energy_capacity: (self.energy_capacity + other.energy_capacity) / 2.0
                + rng.random_range(-context.capacity_range..context.capacity_range),
        };
        child.clamp();
        child
    }
}

/// A complete organism genome containing both neural network and physical traits
#[derive(Clone, Debug, PartialEq, GenerateRandom, RandomlyMutable, Crossover)]
#[randmut(create_context = OrganismCtx)]
#[crossover(with_context = OrganismCtx)]
struct OrganismGenome {
    brain: NeuralNetwork<8, 2>,
    stats: PhysicalStats,
}

/// Running instance of an organism with current position and energy
struct OrganismInstance {
    genome: OrganismGenome,
    x: f32,
    y: f32,
    angle: f32,
    energy: f32,
    lifetime: usize,
    food_eaten: usize,
}

impl OrganismInstance {
    fn new(genome: OrganismGenome) -> Self {
        let energy = genome.stats.energy_capacity;
        Self {
            genome,
            x: rand::random::<f32>() * WORLD_WIDTH,
            y: rand::random::<f32>() * WORLD_HEIGHT,
            angle: rand::random::<f32>() * 2.0 * PI,
            energy,
            lifetime: 0,
            food_eaten: 0,
        }
    }

    /// Simulate one timestep: sense food, decide movement, consume energy, age
    fn step(&mut self, food_sources: &[(f32, f32)]) {
        self.lifetime += 1;

        // find nearest food
        let mut nearest_food_dist = f32::INFINITY;
        let mut nearest_food_angle = 0.0;
        let mut nearest_food_x_diff = 0.0;
        let mut nearest_food_y_diff = 0.0;

        for &(fx, fy) in food_sources {
            let dx = fx - self.x;
            let dy = fy - self.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < self.genome.stats.sense_range && dist < nearest_food_dist {
                nearest_food_dist = dist;
                nearest_food_angle = (dy.atan2(dx) - self.angle).sin();
                nearest_food_x_diff = (dx / 100.0).clamp(-1.0, 1.0);
                nearest_food_y_diff = (dy / 100.0).clamp(-1.0, 1.0);
            }
        }

        let sense_food = if nearest_food_dist < self.genome.stats.sense_range {
            1.0
        } else {
            0.0
        };

        // Create inputs for neural network:
        // 0: current energy level (0-1)
        // 1: food detected (0 or 1)
        // 2: nearest food angle (normalized)
        // 3: nearest food x diff
        // 4: nearest food y diff
        // 5: speed stat (normalized)
        // 6: energy capacity (normalized)
        // 7: age (slow-paced, up to 1 at age 1000)
        let inputs = [
            (self.energy / self.genome.stats.energy_capacity).clamp(0.0, 1.0),
            sense_food,
            nearest_food_angle,
            nearest_food_x_diff,
            nearest_food_y_diff,
            (self.genome.stats.speed / 5.0).clamp(0.0, 1.0),
            (self.genome.stats.energy_capacity / 200.0).clamp(0.0, 1.0),
            (self.lifetime as f32 / 1000.0).clamp(0.0, 1.0),
        ];

        // get movement outputs from neural network
        let outputs = self.genome.brain.predict(inputs);
        let move_forward = (outputs[0] * self.genome.stats.speed).clamp(-5.0, 5.0);
        let turn = (outputs[1] * PI / 4.0).clamp(-PI / 8.0, PI / 8.0);

        // update position and angle
        self.angle += turn;
        self.x += move_forward * self.angle.cos();
        self.y += move_forward * self.angle.sin();

        // wrap around world
        if self.x < 0.0 {
            self.x += WORLD_WIDTH;
        } else if self.x >= WORLD_WIDTH {
            self.x -= WORLD_WIDTH;
        }
        if self.y < 0.0 {
            self.y += WORLD_HEIGHT;
        } else if self.y >= WORLD_HEIGHT {
            self.y -= WORLD_HEIGHT;
        }

        // consume energy for movement
        let movement_cost = (move_forward.abs() / self.genome.stats.speed).max(0.5);
        self.energy -= movement_cost * MOVEMENT_ENERGY_COST;

        // consume energy for existing
        self.energy -= IDLE_ENERGY_COST;
    }

    /// Check if organism lands on food and consume it
    fn eat(&mut self, food_sources: &mut Vec<(f32, f32)>) {
        food_sources.retain(|&(fx, fy)| {
            let dx = fx - self.x;
            let dy = fy - self.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < FOOD_DETECTION_DISTANCE {
                // ate food
                self.energy +=
                    BASE_FOOD_ENERGY + (self.genome.stats.strength * STRENGTH_ENERGY_MULTIPLIER);
                self.energy = self.energy.min(self.genome.stats.energy_capacity);
                self.food_eaten += 1;
                false
            } else {
                true
            }
        });
    }

    fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    fn fitness(&self) -> f32 {
        let food_fitness = (self.food_eaten as f32) * FITNESS_PER_FOOD;
        food_fitness
    }
}

/// Evaluate an organism's fitness by running a simulation
fn evaluate_organism(genome: &OrganismGenome) -> f32 {
    let mut rng = rand::rng();

    let mut food_sources: Vec<(f32, f32)> = (0..INITIAL_FOOD_COUNT)
        .map(|_| {
            (
                rng.random_range(0.0..WORLD_WIDTH),
                rng.random_range(0.0..WORLD_HEIGHT),
            )
        })
        .collect();

    let mut instance = OrganismInstance::new(genome.clone());

    for _ in 0..SIMULATION_TIMESTEPS {
        if instance.is_alive() {
            instance.step(&food_sources);
            instance.eat(&mut food_sources);
        }

        // respawn food
        if food_sources.len() < FOOD_RESPAWN_THRESHOLD {
            food_sources.push((
                rng.random_range(0.0..WORLD_WIDTH),
                rng.random_range(0.0..WORLD_HEIGHT),
            ));
        }
    }

    instance.fitness()
}

fn main() {
    let mut rng = rand::rng();

    println!("Starting genetic NEAT simulation with physical traits");
    println!("Population: {} organisms", POPULATION_SIZE);
    println!("Each has: Neural Network Brain + Physical Stats (Speed, Strength, Sense Range, Energy Capacity)\n");

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, POPULATION_SIZE),
        FitnessEliminator::new_with_default(evaluate_organism),
        CrossoverRepopulator::new(MUTATION_RATE, OrganismCtx::default()),
    );

    for generation in 0..=HIGHEST_GENERATION {
        sim.next_generation();

        let sample = &sim.genomes[0];
        let fitness = evaluate_organism(sample);

        println!(
            "Gen {}: Sample fitness: {:.1} | Speed: {:.2}, Strength: {:.2}, Sense: {:.1}, Capacity: {:.1}",
            generation, fitness, sample.stats.speed, sample.stats.strength, sample.stats.sense_range, sample.stats.energy_capacity
        );
    }

    println!("\nSimulation complete!");
}
