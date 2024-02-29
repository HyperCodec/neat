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
}

impl GenerateRandom for MyAgentDNA {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            network: NeuralNetworkTopology::new(0.01, 3, rng),
        }
    }
}

struct MyAgent {
    network: NeuralNetwork<1, 2>,
    // ... other state
}

impl From<&MyAgentDNA> for MyAgent {
    fn from(value: &MyAgentDNA) -> Self {
        Self {
            network: NeuralNetwork::from(&value.network),
        }
    }
}

fn fitness(dna: &MyAgentDNA) -> f32 {
    // agent will simply try to predict whether a number is greater than 0.5
    let mut agent = MyAgent::from(dna);
    let mut rng = rand::thread_rng();
    let mut fitness = 0;

    // use repeated tests to avoid situational bias and some local maximums, overall providing more accurate score
    for _ in 0..10 {
        let n = rng.gen::<f32>();
        let above = n > 0.5;

        let res = agent.network.predict([n]);
        let resi = res.iter().max_index();

        if resi == 0 ^ above {
            // agent did not guess correctly, punish slightly (too much will hinder exploration)
            fitness -= 0.5;

            continue;
        }

        // agent guessed correctly, they become more fit.
        fitness += 3.;
    }

    fitness
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        division_pruning_nextgen,
    );

    // simulate 100 generations
    for _ in 0..100 {
        sim.next_generation();
    }

    // display fitness results

}
```

### License
This crate falls under the `MIT` license
