use crate::{activation::NeuronScope, *};
use rand::prelude::*;

// no support for tuple structs derive in genetic-rs yet :(
#[derive(Debug, Clone, PartialEq)]
struct Agent(NeuralNetwork<4, 1>);

impl Prunable for Agent {}

impl RandomlyMutable for Agent {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
        self.0.mutate(rate, rng);
    }
}

impl DivisionReproduction for Agent {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        Self(self.0.divide(rng))
    }
}

impl CrossoverReproduction for Agent {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        Self(self.0.crossover(&other.0, rng))
    }
}

impl GenerateRandom for Agent {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self(NeuralNetwork::new(MutationSettings::default(), rng))
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
    let starting_genomes = Vec::gen_random(100);

    let mut sim = GeneticSim::new(starting_genomes, fitness, division_pruning_nextgen);

    sim.perform_generations(100);
}

#[test]
fn crossover() {
    let starting_genomes = Vec::gen_random(100);

    let mut sim = GeneticSim::new(starting_genomes, fitness, crossover_pruning_nextgen);

    sim.perform_generations(100);
}

#[cfg(feature = "serde")]
#[test]
fn serde() {
    let mut rng = rand::thread_rng();
    let net: NeuralNetwork<5, 10> = NeuralNetwork::new(MutationSettings::default(), &mut rng);

    let text = serde_json::to_string(&net).unwrap();

    let net2: NeuralNetwork<5, 10> = serde_json::from_str(&text).unwrap();

    assert_eq!(net, net2);
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

        for j in 0..2 {
            cache.add(
                NeuronLocation::Output(j),
                activation::builtin::sigmoid(cache.get(&hidden_loc) * 0.7),
            );
        }
    }

    assert_eq!(cache.output(), [2.0688455, 2.0688455]);
}

fn small_test_network() -> NeuralNetwork<1, 1> {
    let mut rng = rand::thread_rng();

    let input = Neuron::new(
        vec![
            (NeuronLocation::Hidden(0), 1.),
            (NeuronLocation::Hidden(1), 1.),
            (NeuronLocation::Hidden(2), 1.),
        ],
        NeuronScope::INPUT,
        &mut rng,
    );

    let mut hidden = Neuron::new(
        vec![(NeuronLocation::Output(0), 1.)],
        NeuronScope::HIDDEN,
        &mut rng,
    );
    hidden.input_count = 1;

    let mut output = Neuron::new(vec![], NeuronScope::OUTPUT, &mut rng);
    output.input_count = 3;

    NeuralNetwork {
        input_layer: [input],
        hidden_layers: vec![hidden; 3],
        output_layer: [output],
        mutation_settings: MutationSettings::default(),
        total_connections: 6,
    }
}

#[test]
fn remove_neuron() {
    let mut network = small_test_network();

    network.remove_neuron(NeuronLocation::Hidden(1));

    assert_eq!(network.total_connections, 4);

    let expected = vec![NeuronLocation::Hidden(0), NeuronLocation::Hidden(1)];
    let got: Vec<_> = network.input_layer[0].outputs.iter().map(|c| c.0).collect();

    assert_eq!(got, expected);
}

#[test]
fn recalculate_connections() {
    let mut rng = rand::thread_rng();

    let input = Neuron::new(
        vec![
            (NeuronLocation::Hidden(0), 1.),
            (NeuronLocation::Hidden(1), 1.),
            (NeuronLocation::Hidden(2), 1.),
        ],
        NeuronScope::INPUT,
        &mut rng,
    );

    let hidden = Neuron::new(
        vec![(NeuronLocation::Output(0), 1.)],
        NeuronScope::HIDDEN,
        &mut rng,
    );

    let output = Neuron::new(vec![], NeuronScope::OUTPUT, &mut rng);

    let mut network = NeuralNetwork {
        input_layer: [input],
        hidden_layers: vec![hidden; 3],
        output_layer: [output],
        mutation_settings: MutationSettings::default(),
        total_connections: 0,
    };

    network.recalculate_connections();

    assert_eq!(network.total_connections, 6);

    for n in &network.hidden_layers {
        assert_eq!(n.input_count, 1);
    }

    assert_eq!(network.output_layer[0].input_count, 3);
}

#[test]
fn add_connection() {
    let mut network = small_test_network();

    assert!(network.add_connection(
        Connection {
            from: NeuronLocation::Hidden(0),
            to: NeuronLocation::Hidden(1),
        },
        1.
    ));

    assert_eq!(network.total_connections, 7);
    assert_eq!(network.hidden_layers[1].input_count, 2);

    assert!(!network.add_connection(
        Connection {
            from: NeuronLocation::Hidden(1),
            to: NeuronLocation::Hidden(0)
        },
        1.
    ));

    assert_eq!(network.total_connections, 7);

    assert!(network.add_connection(
        Connection {
            from: NeuronLocation::Hidden(1),
            to: NeuronLocation::Hidden(2),
        },
        1.
    ));

    assert!(!network.add_connection(
        Connection {
            from: NeuronLocation::Hidden(2),
            to: NeuronLocation::Hidden(0),
        },
        1.
    ));
}

#[test]
fn remove_connection() {
    let mut network = small_test_network();

    assert!(!network.remove_connection(Connection {
        from: NeuronLocation::Hidden(0),
        to: NeuronLocation::Output(0),
    }));

    assert_eq!(network.total_connections, 5);

    assert!(network.remove_connection(Connection {
        from: NeuronLocation::Input(0),
        to: NeuronLocation::Hidden(1),
    }));

    assert_eq!(network.total_connections, 3);
    assert_eq!(network.hidden_layers.len(), 2);
}

#[test]
fn random_location_in_scope() {
    let mut rng = rand::thread_rng();
    let mut network = small_test_network();

    assert_eq!(
        network.random_location_in_scope(&mut rng, NeuronScope::INPUT),
        Some(NeuronLocation::Input(0))
    );

    // TODO `assert_matches` when it is stable
    assert!(matches!(
        network.random_location_in_scope(&mut rng, NeuronScope::HIDDEN),
        Some(NeuronLocation::Hidden(_))
    ));

    let multi = network.random_location_in_scope(&mut rng, !NeuronScope::INPUT);
    assert!(
        matches!(multi, Some(NeuronLocation::Hidden(_)))
            || matches!(multi, Some(NeuronLocation::Output(_)))
    );

    network.hidden_layers = vec![];
    assert!(network
        .random_location_in_scope(&mut rng, NeuronScope::HIDDEN)
        .is_none());
}

#[test]
fn split_connection() {
    let mut rng = rand::thread_rng();
    let mut network = small_test_network();

    network.split_connection(Connection {
        from: NeuronLocation::Input(0),
        to: NeuronLocation::Hidden(1),
    }, &mut rng);

    assert_eq!(network.total_connections, 7);

    let n = &network.hidden_layers[3];
    assert_eq!(n.outputs[0].0, NeuronLocation::Hidden(1));
    assert_eq!(n.input_count, 1);
}

// TODO test every method