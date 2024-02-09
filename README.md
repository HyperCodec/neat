# neat
Implementation of the NEAT algorithm using `genetic-rs`

### Features
- rayon - Uses parallelization on the `NeuralNetwork` struct and adds the `rayon` feature to the `genetic-rs` re-export.

### How To Use
When working with this crate, you'll want to use the `NeuralNetworkTopology` struct in your agent's DNA and
the use `NeuralNetwork::from` when you finally want to test its performance. The `genetic-rs` crate is also re-exported with the rest of this crate.

Here's an example of how one might use this crate:
```rust
use neat::*;

#[derive(Clone)]
struct MyAgentDNA {
    network: NeuralNetworkTopology<1, 2>,
    other_stuff: Foo,
}

impl RandomlyMutable for MyAgentDNA {
    fn mutate(&mut self,  rate: f32, rng: &mut impl rand::Rng) {
        self.network.mutate(rate, rng);
        self.other_stuff.mutate(rate, rng);
    }
}

impl DivisionReproduction for MyAgentDNA {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        Self {
            network: self.network.spawn_child(rng),
            // ...
        }
    }
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