use std::sync::atomic::{AtomicBool, AtomicUsize};

use crate::*;
use atomic_float::AtomicF32;
use rand::prelude::*;

// no support for tuple structs derive in genetic-rs yet :(
#[derive(Debug, Clone)]
struct Agent(NeuralNetwork<4, 1>);

impl Prunable for Agent {}

impl RandomlyMutable for Agent {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.0.mutate(rate, rng);
    }
}

impl DivisionReproduction for Agent {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        Self(self.0.divide(rng))
    }
}

struct GuessTheNumber(f32);

impl GuessTheNumber {
    fn new(rng: &mut impl Rng) -> Self {
        Self(rng.gen())
    }

    fn guess(&self, n: f32) -> Option<f32> {
        if n > self.0 + 1.0e-5 {
            return Some(1.);
        }

        if n < self.0 - 1.0e-5 {
            return Some(-1.);
        }

        // guess was correct (or at least within margin of error).
        None
    }
}

fn fitness(agent: &Agent) -> f32 {
    let mut rng = rand::thread_rng();

    let mut fitness = 0.;

    // 10 games for consistency
    for _ in 0..10 {
        let game = GuessTheNumber::new(&mut rng);

        let mut last_guess = 0.;
        let mut last_result = 0.;

        let mut last_guess_2 = 0.;
        let mut last_result_2 = 0.;

        let mut steps = 0;
        loop {
            if steps >= 20 {
                // took too many guesses
                fitness -= 50.;
                break;
            }

            let [cur_guess] =
                agent
                    .0
                    .predict([last_guess, last_result, last_guess_2, last_result_2]);
            let cur_result = game.guess(cur_guess);

            if let Some(result) = cur_result {
                last_guess = last_guess_2;
                last_result = last_result_2;

                last_guess_2 = cur_guess;
                last_result_2 = result;

                fitness -= 1.;
                steps += 1;

                continue;
            }

            fitness += 50.;
            break;
        }
    }

    fitness
}

#[test]
fn division() {
    let mut rng = rand::thread_rng();

    let starting_genomes = (0..100)
        .map(|_| Agent(NeuralNetwork::new(MutationSettings::default(), &mut rng)))
        .collect();

    let mut sim = GeneticSim::new(starting_genomes, fitness, division_pruning_nextgen);

    sim.perform_generations(100);
}

#[test]
fn neural_net_cache_sync() {
    let cache = NeuralNetCache {
        input_layer: [NeuronCache::new(0.3, 0), NeuronCache::new(0.25, 0)],
        hidden_layers: vec![
            NeuronCache::new(0.2, 2),
            NeuronCache::new(0.0, 2),
            NeuronCache::new(1.5, 2),
        ],
        output_layer: [NeuronCache::new(0.0, 3), NeuronCache::new(0.0, 3)],
    };

    for i in 0..2 {
        let input_loc = NeuronLocation::Input(i);

        assert!(cache.claim(&input_loc));

        for j in 0..3 {
            cache.add(
                NeuronLocation::Hidden(j),
                f32::tanh(cache.get(&input_loc) * 1.2),
            );
        }
    }

    for i in 0..3 {
        let hidden_loc = NeuronLocation::Hidden(i);

        assert!(cache.is_ready(&hidden_loc));
        assert!(cache.claim(&hidden_loc));

        for j in 0..2 {
            cache.add(
                NeuronLocation::Output(j),
                activation::builtin::sigmoid(cache.get(&hidden_loc) * 0.7),
            );
        }
    }

    assert_eq!(cache.output(), [2.0688455, 2.0688455]);
}
