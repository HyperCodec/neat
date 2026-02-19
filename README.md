# neat
[<img alt="github" src="https://img.shields.io/github/last-commit/hypercodec/neat" height="20">](https://github.com/hypercodec/neat)
[<img alt="crates.io" src="https://img.shields.io/crates/d/neat" height="20">](https://crates.io/crates/neat)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/neat" height="20">](https://docs.rs/neat)

Implementation of the NEAT algorithm using `genetic-rs`.

### Features
- serde - Implements `Serialize` and `Deserialize` on most of the types in this crate.

*Do you like this crate and want to support it? If so, leave a ⭐*

# How To Use
The `NeuralNetwork<I, O>` struct is the main type exported by this crate. The `I` is the number of input neurons, and `O` is the number of output neurons. It implements `GenerateRandom`, `RandomlyMutable`, `Mitosis`, and `Crossover`, with a lot of customizability. This means that you can use it standalone as your organism's entire genome:
```rust
use neat::*;

fn fitness(net: &NeuralNetwork<5, 6>) -> f32 {
    // ideally you'd test multiple times for consistency,
    // but this is just a simple example.
    // it's also generally good to normalize your inputs between -1..1,
    // but NEAT is usually flexible enough to still work anyways
    let inputs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let outputs = net.predict(inputs);

    // simple fitness: sum of outputs
    // you should replace this with a real fitness test
    outputs.iter().sum()
}

fn main() {
    let mut rng = rand::rng();
    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_without_observer(fitness),
        CrossoverRepopulator::new(0.25, ReproductionSettings::default()),
    );

    sim.perform_generations(100);
}
```

Or just a part of a more complex genome:
```rust,ignore
use neat::*;

#[derive(Clone, Debug)]
struct PhysicalStats {
    strength: f32,
    speed: f32,
    // ...
}

// ... implement `RandomlyMutable`, `GenerateRandom`, `Crossover`, `Default`, etc.

#[derive(Clone, Debug, GenerateRandom, RandomlyMutable, Mitosis, Crossover)]
#[randmut(create_context = MyGenomeCtx)]
#[crossover(with_context = MyGenomeCtx)]
struct MyGenome {
    brain: NeuralNetwork<4, 2>,
    stats: PhysicalStats,
}

impl Default for MyGenomeCtx {
    fn default() -> Self {
        Self {
            brain: ReproductionSettings::default(),
            stats: PhysicalStats::default(),
        }
    }
}

fn fitness(genome: &MyGenome) -> f32 {
    let inputs = [1.0, 2.0, 3.0, 4.0];
    let outputs = genome.brain.predict(inputs);
    // fitness uses both brain output and stats
    outputs.iter().sum::<f32>() + genome.stats.strength + genome.stats.speed
}

// main is the exact same as before
fn main() {
    let mut rng = rand::rng();
    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_without_observer(fitness),
        CrossoverRepopulator::new(0.25, MyGenomeCtx::default()),
    );

    sim.perform_generations(100);
}
```

If you want more in-depth examples, look at the [examples](https://github.com/HyperCodec/neat/tree/main/examples). You can also check out the [genetic-rs docs](https://docs.rs/genetic_rs) to see what other options you have to customize your genetic simulation.

### License
This crate falls under the `MIT` license
