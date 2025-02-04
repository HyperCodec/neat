# neat
[<img alt="github" src="https://img.shields.io/github/last-commit/hypercodec/neat" height="20">](https://github.com/hypercodec/neat)
[<img alt="crates.io" src="https://img.shields.io/crates/d/neat" height="20">](https://crates.io/crates/neat)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/neat" height="20">](https://docs.rs/neat)

Implementation of the NEAT algorithm using `genetic-rs`.

### Features
- serde - Implements `Serialize` and `Deserialize` on most of the types in this crate.

*Do you like this crate and want to support it? If so, leave a ⭐*

# Guide
A neural network has two const generic parameters, `I` and `O`. `I` represents the number of input neurons and `O` represents the number of output neurons. To create a neural network, use `NeuralNetwork::new`:

```rust
use neat::*;

let mut rng = rand::thread_rng();

// creates a randomized neural network with 3 input neurons and 2 output neurons.
let net: NeuralNetwork<3, 2> = NeuralNetwork::new(MutationSettings::default(), &mut rng);
```

Once you have a neural network, you can use it to predict things:

```rust
let prediction = net.predict([1, 2, 3]);
dbg!(prediction);
```

A completely random neural network isn't quite useful, however, so you must run a simulation to train and perfect these networks. Let's look at the following code:

```rust
use neat::*;

// derive some traits so that we can use this agent with `genetic-rs`.
#[derive(Debug, Clone, Prunable, CrossoverReproduction, RandomlyMutable)]
struct MyAgentGenome {
    brain: NeuralNetwork<3, 2>
}

impl GenerateRandom for MyAgentGenome {
    // allows us to use `Vec::gen_random` for the initial population.
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self(NeuralNetwork::new(MutationSettings::default(), rng))
    }
}

// creates a bigger number within the bounds of 0 to 1 as `actual` approaches `expected`.
fn inverse_error(expected: f32, actual: f32) -> f32 {
    1.0 / (1.0 + (expected - actual).abs())
}

fn fitness(agent: &MyAgentGenome) -> f32 {
    let mut rng = rand::thread_rng();
    let mut fit = 0;

    for _ in 0..10 {
        // run the test multiple times for consistency

        let inputs = [rng.gen(), rng.gen(), rng.gen()];
        
        // try to force the network to learn to do some basic logic
        let expected0 = (inputs[0] >= 0.5 && inputs[1] < 0.5) as f32;
        let expected1 = (inputs[2] >= 0.5) as f32;

        let output = agent.brain.predict(inputs);

        fit += inverse_error(expected0, output[0]);
        fit += inverse_error(expected1, output[1]);
    }

    fit
}

fn main() {
    let mut sim = GeneticSim::new(
        // create a population of 100 random neural networks
        Vec::gen_random(100),

        // provide the fitness function that will
        // test the agents individually so the nextgen
        // function can eliminate the weaker ones.
        fitness,

        // this nextgen function will kill/drop agents 
        // that don't have a high enough fitness, and repopulate
        // by performing crossover reproduction between the remaining ones
        crossover_pruning_nextgen,
    );

    // fast forward 100 generations. identical to looping 
    // 100 times with `sim.next_generation()`.
    sim.perform_generations(100);
}
```

The struct `MyAgentGenome` is created to wrap the `NeuralNetwork` and functions as the overall hereditary data of an agent. In a more complex scenario, you could add more `genetic-rs`-compatible types to store other hereditary information, such as an agent's size or speed.

Check out the [examples](https://github.com/HyperCodec/neat/tree/main/examples) for more usecases.

### License
This crate falls under the `MIT` license
