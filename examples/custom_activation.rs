//! An example implementation of a custom activation function.

use neat::*;
use rand::prelude::*;

#[derive(DivisionReproduction, RandomlyMutable, Clone)]
struct AgentDNA {
    network: NeuralNetworkTopology<2, 2>,
}

impl Prunable for AgentDNA {}

impl GenerateRandom for AgentDNA {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            network: NeuralNetworkTopology::new(0.01, 3, rng),
        }
    }
}

fn fitness(g: &AgentDNA) -> f32 {
    let network: NeuralNetwork<2, 2> = NeuralNetwork::from(&g.network);
    let mut fitness = 0.;
    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let n = rng.gen::<f32>();
        let n2 = rng.gen::<f32>();

        let expected = if (n + n2) / 2. >= 0.5 { 0 } else { 1 };

        let result = network.predict([n, n2]);
        network.flush_state();

        // partial_cmp chance of returning None in this smh
        let result = result.iter().max_index();

        if result == expected {
            fitness += 1.;
        } else {
            fitness -= 1.;
        }
    }

    fitness
}

#[cfg(feature = "serde")]
fn serde_nextgen(rewards: Vec<(AgentDNA, f32)>) -> Vec<AgentDNA> {
    let max = rewards
        .iter()
        .max_by(|(_, ra), (_, rb)| ra.total_cmp(rb))
        .unwrap();

    let ser = NNTSerde::from(&max.0.network);
    let data = serde_json::to_string_pretty(&ser).unwrap();
    std::fs::write("best-agent.json", data).expect("Failed to write to file");

    division_pruning_nextgen(rewards)
}

fn main() {
    let sin_activation = activation_fn!(f32::sin);
    register_activation(sin_activation);

    #[cfg(not(feature = "rayon"))]
    let mut rng = rand::thread_rng();

    let mut sim = GeneticSim::new(
        #[cfg(not(feature = "rayon"))]
        Vec::gen_random(&mut rng, 100),
        #[cfg(feature = "rayon")]
        Vec::gen_random(100),
        fitness,
        #[cfg(not(feature = "serde"))]
        division_pruning_nextgen,
        #[cfg(feature = "serde")]
        serde_nextgen,
    );

    for _ in 0..200 {
        sim.next_generation();
    }
}
