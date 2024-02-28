# neat
[<img alt="github" src="https://img.shields.io/github/last-commit/inflectrix/neat" height="20">](https://github.com/inflectrix/neat)
[<img alt="crates.io" src="https://img.shields.io/crates/d/neat" height="20">](https://crates.io/crates/neat)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/neat" height="20">](https://docs.rs/neat)

Implementation of the NEAT algorithm using `genetic-rs`.

### Features
- rayon - Uses parallelization on the `NeuralNetwork` struct and adds the `rayon` feature to the `genetic-rs` re-export.
- serde - Adds the NNTSerde struct and allows for serialization of `NeuralNetworkTopology`
- crossover - Implements the `CrossoverReproduction` trait on `NeuralNetworkTopology` and adds the `crossover` feature to the `genetic-rs` re-export.

*Do you like this repo and want to support it? If so, leave a ⭐*

### How To Use
When working with this crate, you'll want to use the `NeuralNetworkTopology` struct in your agent's DNA and
the use `NeuralNetwork::from` when you finally want to test its performance. The `genetic-rs` crate is also re-exported with the rest of this crate.

Here's an example of how one might use this crate:
```rust
use neat::*;

#[derive(Clone, RandomlyMutable, DivisionReproduction)]
struct MyAgentDNA {
    network: NeuralNetworkTopology<1, 2>,
    other_stuff: Foo,
}

impl GenerateRandom for MyAgentDNA {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            network: NeuralNetworkTopology::new(0.01, 3, rng),
            other_stuff: Foo::gen_random(rng),
        }
    }
}

struct MyAgent {
    network: NeuralNetwork<1, 2>,
    some_other_state: Bar,
}

impl From<&MyAgentDNA> for MyAgent {
    fn from(value: &MyAgentDNA) -> Self {
        Self {
            network: NeuralNetwork::from(&value.network),
            some_other_state: Bar::default(),
        }
    }
}

fn fitness(dna: &MyAgentDNA) -> f32 {
    let mut agent = MyAgent::from(dna);

    // ... use agent.network.predict() and agent.network.flush() throughout multiple iterations
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        division_pruning_nextgen,
    );

    // ... simulate generations, etc.
}
```

### License
This crate falls under the `MIT` license
