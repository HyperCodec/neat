use std::collections::HashMap;

use crate::{activation::builtin::linear_activation, *};
use genetic_rs::prelude::rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GraphCheckState {
    CurrentCycle,
    Checked,
}

fn assert_graph_invariants<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) {
    let mut visited = HashMap::new();

    for i in 0..I {
        dfs(net, NeuronLocation::Input(i), &mut visited);
    }

    for i in 0..net.hidden_layers.len() {
        let loc = NeuronLocation::Hidden(i);
        if !visited.contains_key(&loc) {
            panic!("hanging neuron: {loc:?}");
        }
    }
}

// simple colored dfs for checking graph invariants.
fn dfs<const I: usize, const O: usize>(
    net: &NeuralNetwork<I, O>,
    loc: NeuronLocation,
    visited: &mut HashMap<NeuronLocation, GraphCheckState>,
) {
    if let Some(existing) = visited.get(&loc) {
        match *existing {
            GraphCheckState::CurrentCycle => panic!("cycle detected on {loc:?}"),
            GraphCheckState::Checked => return,
        }
    }

    visited.insert(loc, GraphCheckState::CurrentCycle);

    for loc2 in net[loc].outputs.keys() {
        dfs(net, *loc2, visited);
    }

    visited.insert(loc, GraphCheckState::Checked);
}

struct InputCountsCache<const O: usize> {
    hidden_layers: Vec<usize>,
    output: [usize; O],
}

impl<const O: usize> InputCountsCache<O> {
    fn tally(&mut self, loc: NeuronLocation) {
        match loc {
            NeuronLocation::Input(_) => panic!("input neurons can't have inputs"),
            NeuronLocation::Hidden(i) => self.hidden_layers[i] += 1,
            NeuronLocation::Output(i) => self.output[i] += 1,
        }
    }
}

// asserts that cached/tracked values are correct. mainly only used for
// input count and such
fn assert_cache_consistency<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) {
    let mut cache = InputCountsCache {
        hidden_layers: vec![0; net.hidden_layers.len()],
        output: [0; O],
    };

    for i in 0..I {
        let n = &net[NeuronLocation::Input(i)];
        for loc in n.outputs.keys() {
            cache.tally(*loc);
        }
    }

    for n in &net.hidden_layers {
        for loc in n.outputs.keys() {
            cache.tally(*loc);
        }
    }

    for (i, x) in cache.hidden_layers.into_iter().enumerate() {
        if x == 0 {
            // redundant because of graph invariants, but better safe than sorry
            panic!("found hanging neuron");
        }

        assert_eq!(x, net.hidden_layers[i].input_count);
    }

    for (i, x) in cache.output.into_iter().enumerate() {
        assert_eq!(x, net.output_layer[i].input_count);
    }
}

fn assert_network_invariants<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) {
    assert_graph_invariants(net);
    assert_cache_consistency(net);
    // TODO other invariants
}

const TEST_COUNT: u64 = 1000;
fn rng_test(test: impl Fn(&mut StdRng) + Sync) {
    (0..TEST_COUNT).into_par_iter().for_each(|seed| {
        let mut rng = StdRng::seed_from_u64(seed);
        test(&mut rng);
    });
}

#[test]
fn create_network() {
    rng_test(|rng| {
        let net = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net);
    });
}

#[test]
fn split_connection() {
    // rng doesn't matter here since it's just adding bias in eval
    let mut rng = StdRng::seed_from_u64(0xabcdef);

    let mut net = NeuralNetwork::<1, 1>::new(&mut rng);
    assert_network_invariants(&net);

    net.split_connection(
        Connection {
            from: NeuronLocation::Input(0),
            to: NeuronLocation::Output(0),
        },
        &mut rng,
    );
    assert_network_invariants(&net);

    assert_eq!(
        *net.input_layer[0].outputs.keys().next().unwrap(),
        NeuronLocation::Hidden(0)
    );
    assert_eq!(
        *net.hidden_layers[0].outputs.keys().next().unwrap(),
        NeuronLocation::Output(0)
    );
}

#[test]
fn add_connection() {
    let mut rng = StdRng::seed_from_u64(0xabcdef);
    let mut net = NeuralNetwork {
        input_layer: [Neuron::new_with_activation(
            HashMap::new(),
            activation_fn!(linear_activation),
            &mut rng,
        )],
        hidden_layers: vec![],
        output_layer: [Neuron::new_with_activation(
            HashMap::new(),
            activation_fn!(linear_activation),
            &mut rng,
        )],
    };
    assert_network_invariants(&net);

    let mut conn = Connection {
        from: NeuronLocation::Input(0),
        to: NeuronLocation::Output(0),
    };
    assert!(net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    assert!(!net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    let mut outputs = HashMap::new();
    outputs.insert(NeuronLocation::Output(0), 0.1);
    let n = Neuron::new_with_activation(outputs, activation_fn!(linear_activation), &mut rng);

    net.add_neuron(n.clone());
    // temporarily broken invariants bc of hanging neuron

    conn.to = NeuronLocation::Hidden(0);
    assert!(net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    net.add_neuron(n);

    conn.to = NeuronLocation::Hidden(1);
    assert!(net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    conn.from = NeuronLocation::Hidden(0);
    assert!(net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    net.split_connection(conn, &mut rng);
    assert_network_invariants(&net);

    conn.from = NeuronLocation::Hidden(2);
    conn.to = NeuronLocation::Hidden(0);

    assert!(!net.add_connection(conn, 0.1));
    assert_network_invariants(&net);

    // random stress testing
    rng_test(|rng| {
        let mut net = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net);
        for _ in 0..50 {
            net.add_random_connection(10, rng);
            assert_network_invariants(&net);
        }
    });
}

#[test]
fn remove_connection() {
    let mut rng = StdRng::seed_from_u64(0xabcdef);
    let mut net = NeuralNetwork {
        input_layer: [Neuron::new_with_activation(
            HashMap::from([
                (NeuronLocation::Output(0), 0.1),
                (NeuronLocation::Hidden(0), 1.0),
            ]),
            activation_fn!(linear_activation),
            &mut rng,
        )],
        hidden_layers: vec![Neuron {
            input_count: 1,
            outputs: HashMap::new(), // not sure whether i want neurons with no outputs to break the invariant/be removed
            bias: 0.0,
            activation_fn: activation_fn!(linear_activation),
        }],
        output_layer: [Neuron {
            input_count: 1,
            outputs: HashMap::new(),
            bias: 0.0,
            activation_fn: activation_fn!(linear_activation),
        }],
    };
    assert_network_invariants(&net);

    assert!(!net.remove_connection(Connection {
        from: NeuronLocation::Input(0),
        to: NeuronLocation::Output(0)
    }));
    assert_network_invariants(&net);

    assert!(net.remove_connection(Connection {
        from: NeuronLocation::Input(0),
        to: NeuronLocation::Hidden(0)
    }));
    assert_network_invariants(&net);

    rng_test(|rng| {
        let mut net = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net);

        for _ in 0..70 {
            net.add_random_connection(10, rng);
            assert_network_invariants(&net);

            if rng.random_bool(0.25) {
                // rng allows network to form more complex edge cases.
                net.remove_random_connection(5, rng);
                // don't need to remove neuron since this
                // method handles it automatically.
                assert_network_invariants(&net);
            }
        }
    });
}

// TODO remove_neuron test

#[test]
fn predict_basic() {
    // Build a minimal 1-in / 1-out network with linear activations and zero bias
    // so the output is exactly: input * weight.
    let weight = 0.5_f32;
    let net = NeuralNetwork {
        input_layer: [Neuron {
            input_count: 0,
            outputs: HashMap::from([(NeuronLocation::Output(0), weight)]),
            bias: 0.0,
            activation_fn: activation_fn!(linear_activation),
        }],
        hidden_layers: vec![],
        output_layer: [Neuron {
            input_count: 1,
            outputs: HashMap::new(),
            bias: 0.0,
            activation_fn: activation_fn!(linear_activation),
        }],
    };

    let inputs = [2.0_f32];
    let outputs = net.predict(inputs);
    let expected = inputs[0] * weight;
    assert!(
        (outputs[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        outputs[0]
    );

    // Zero input should yield zero output (no bias).
    let outputs_zero = net.predict([0.0]);
    assert!(
        outputs_zero[0].abs() < 1e-5,
        "expected 0.0, got {}",
        outputs_zero[0]
    );

    // Stress-test with random networks using default (sigmoid) output activations.
    // Use a sequential loop to avoid nested rayon parallelism (predict uses rayon internally).
    for seed in 0..TEST_COUNT {
        let mut rng = StdRng::seed_from_u64(seed);
        let net = NeuralNetwork::<5, 5>::new(&mut rng);
        let inputs = [0.1, 0.2, 0.3, 0.4, 0.5];
        let outputs = net.predict(inputs);
        // sigmoid outputs are in the open interval (0, 1)
        for &v in &outputs {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {v} out of range (0, 1)");
        }
    }
}

#[test]
fn predict_consistency() {
    // Repeated calls with the same inputs must return results within floating-point
    // tolerance. Exact equality is not guaranteed because the parallel atomic
    // accumulation order may vary between runs.
    // Use a sequential loop to avoid nested rayon parallelism (predict uses rayon internally).
    for seed in 0..TEST_COUNT {
        let mut rng = StdRng::seed_from_u64(seed);
        let net = NeuralNetwork::<5, 3>::new(&mut rng);
        let inputs = [1.0, -1.0, 0.5, 0.0, -0.5];
        let first = net.predict(inputs);
        for _ in 0..5 {
            let result = net.predict(inputs);
            for (a, b) in first.iter().zip(result.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "predict returned inconsistent results: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn predict_parallel_no_deadlock() {
    // Build a network with a more complex topology via mutation, then run many
    // parallel predictions to verify that the internal parallel evaluation path
    // completes without deadlocks or race conditions.
    let mut rng = StdRng::seed_from_u64(0xdeadbeef);
    let settings = MutationSettings::default();
    let mut net = NeuralNetwork::<4, 2>::new(&mut rng);
    for _ in 0..20 {
        net.mutate(&settings, 0.5, &mut rng);
    }

    let results: Vec<[f32; 2]> = (0..100_u32)
        .into_par_iter()
        .map(|i| {
            let inputs = [i as f32 * 0.01, 0.5, -0.3, 1.0];
            net.predict(inputs)
        })
        .collect();

    // All outputs should be finite (no NaN / Inf from race conditions).
    for outputs in &results {
        for &v in outputs {
            assert!(v.is_finite(), "non-finite output {v} detected");
        }
    }
}

const NUM_MUTATIONS: usize = 50;
const MUTATION_RATE: f32 = 0.25;
#[test]
fn mutate() {
    rng_test(|rng| {
        let mut net = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net);

        let settings = MutationSettings::default();

        for _ in 0..NUM_MUTATIONS {
            net.mutate(&settings, MUTATION_RATE, rng);
            assert_network_invariants(&net);
        }
    });
}

#[test]
fn crossover() {
    rng_test(|rng| {
        let mut net1 = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net1);

        let mut net2 = NeuralNetwork::<10, 10>::new(rng);
        assert_network_invariants(&net2);

        let settings = ReproductionSettings::default();

        for _ in 0..NUM_MUTATIONS {
            let a = net1.crossover(&net2, &settings, MUTATION_RATE, rng);
            assert_network_invariants(&a);

            let b = net2.crossover(&net1, &settings, MUTATION_RATE, rng);
            assert_network_invariants(&b);

            net1 = a;
            net2 = b;
        }
    });
}

#[cfg(feature = "serde")]
mod serde {
    use super::rng_test;
    use crate::*;

    #[test]
    fn full_serde() {
        rng_test(|rng| {
            let net1 = NeuralNetwork::<10, 10>::new(rng);

            let mut buf = Vec::new();
            let writer = std::io::Cursor::new(&mut buf);
            let mut serializer = serde_json::Serializer::new(writer);

            serde_path_to_error::serialize(&net1, &mut serializer).unwrap();
            let serialized = serde_json::to_string(&net1).unwrap();
            let net2: NeuralNetwork<10, 10> = serde_json::from_str(&serialized).unwrap();
            assert_eq!(net1, net2);
        });
    }
}
