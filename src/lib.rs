//! A simple crate that implements the Neuroevolution Augmenting Topologies algorithm using [genetic-rs](https://crates.io/crates/genetic-rs)
//! ### Feature Roadmap:
//! - [x] base (single-core) crate
//! - [x] rayon
//! - [x] serde
//! - [x] crossover
//!
//! You can get started by looking at [genetic-rs docs](https://docs.rs/genetic-rs) and checking the examples for this crate.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// A module containing the [`NeuralNetworkTopology`] struct. This is what you want to use in the DNA of your agent, as it is the thing that goes through nextgens and suppors mutation.
pub mod topology;

/// A module containing the main [`NeuralNetwork`] struct.
/// This has state/cache and will run the predictions. Make sure to run [`NeuralNetwork::flush_state`] between uses of [`NeuralNetwork::predict`].
pub mod runnable;

pub use genetic_rs::prelude::*;
pub use runnable::*;
pub use topology::*;

#[cfg(feature = "serde")]
pub use nnt_serde::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[derive(RandomlyMutable, DivisionReproduction, Clone)]
    struct AgentDNA {
        network: NeuralNetworkTopology<2, 1>,
    }

    impl Prunable for AgentDNA {}

    impl GenerateRandom for AgentDNA {
        fn gen_random(rng: &mut impl Rng) -> Self {
            Self {
                network: NeuralNetworkTopology::new(0.01, 3, rng),
            }
        }
    }

    #[test]
    fn basic_test() {
        let fitness = |g: &AgentDNA| {
            let network = NeuralNetwork::from(&g.network);
            let mut fitness = 0.;
            let mut rng = rand::thread_rng();

            for _ in 0..100 {
                let n = rng.gen::<f32>() * 10000.;
                let base = rng.gen::<f32>() * 10.;
                let expected = n.log(base);

                let [answer] = network.predict([n, base]);
                network.flush_state();

                fitness += 5. / (answer - expected).abs();
            }

            fitness
        };

        #[cfg(not(feature = "rayon"))]
        let mut rng = rand::thread_rng();

        let mut sim = GeneticSim::new(
            #[cfg(not(feature = "rayon"))]
            Vec::gen_random(&mut rng, 100),
            #[cfg(feature = "rayon")]
            Vec::gen_random(100),
            fitness,
            division_pruning_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        let mut fits: Vec<_> = sim.genomes.iter().map(fitness).collect();

        fits.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        dbg!(fits);
    }
}
