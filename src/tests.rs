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
fn add_connection_converging_paths() {
    // Validates that is_connection_safe allows "diamond" DAG topologies where a single
    // node has two outgoing paths that both reach the same downstream neuron:
    //   Hidden(0) -> Output(0)               (direct)
    //   Hidden(0) -> Hidden(1) -> Output(0)  (indirect)
    // Adding Input(0) -> Hidden(0) is safe (no cycle); the original dfs falsely
    // rejected it by treating the second visit to Output(0) as a cycle.
    let mut rng = StdRng::seed_from_u64(0xabcdef);

    let mut net = NeuralNetwork {
        input_layer: [Neuron::new_with_activation(
            HashMap::new(),
            activation_fn!(linear_activation),
            &mut rng,
        )],
        hidden_layers: vec![
            Neuron::new_with_activation(
                HashMap::new(),
                activation_fn!(linear_activation),
                &mut rng,
            ),
            Neuron::new_with_activation(
                HashMap::new(),
                activation_fn!(linear_activation),
                &mut rng,
            ),
        ],
        output_layer: [Neuron::new_with_activation(
            HashMap::new(),
            activation_fn!(linear_activation),
            &mut rng,
        )],
    };

    // Build the diamond: Hidden(0) -> Output(0) and Hidden(0) -> Hidden(1) -> Output(0)
    assert!(net.add_connection(
        Connection {
            from: NeuronLocation::Hidden(0),
            to: NeuronLocation::Output(0)
        },
        1.0
    ));
    assert!(net.add_connection(
        Connection {
            from: NeuronLocation::Hidden(1),
            to: NeuronLocation::Output(0)
        },
        1.0
    ));
    assert!(net.add_connection(
        Connection {
            from: NeuronLocation::Hidden(0),
            to: NeuronLocation::Hidden(1)
        },
        1.0
    ));

    // Input(0) -> Hidden(0) is safe (no cycle), but the original dfs falsely rejected it
    // because traversing from Hidden(0) visits Output(0) via the direct path first, then
    // encounters Output(0) again via Hidden(1), treating the revisit as a cycle.
    assert!(net.add_connection(
        Connection {
            from: NeuronLocation::Input(0),
            to: NeuronLocation::Hidden(0)
        },
        1.0
    ));

    assert_network_invariants(&net);
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
    // build a minimal 1-in / 1-out network with linear activations and zero bias
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

    // zero input should yield zero output (no bias).
    let outputs_zero = net.predict([0.0]);
    assert!(
        outputs_zero[0].abs() < 1e-5,
        "expected 0.0, got {}",
        outputs_zero[0]
    );

    // stress-test with random networks using default (sigmoid) output activations.
    // use a sequential loop to avoid nested rayon parallelism (predict uses rayon internally).
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
    // repeated calls with the same inputs must return results within floating-point
    // tolerance. exact equality is not guaranteed because the parallel atomic
    // accumulation order may vary between runs.
    // use a sequential loop to avoid nested rayon parallelism (predict uses rayon internally).
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
    // build a network with a more complex topology via mutation, then run many
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

    // all outputs should be finite (no NaN / inf from race conditions).
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

#[allow(dead_code)]
fn find_cycle_helper<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) -> Option<Vec<NeuronLocation>> {
    use std::collections::HashMap as HM;
    fn dfs<const I: usize, const O: usize>(
        net: &NeuralNetwork<I, O>,
        loc: NeuronLocation,
        visited: &mut HM<NeuronLocation, bool>,
        path: &mut Vec<NeuronLocation>,
    ) -> Option<Vec<NeuronLocation>> {
        if let Some(&in_progress) = visited.get(&loc) {
            if in_progress {
                let s = path.iter().position(|&x| x == loc).unwrap();
                return Some(path[s..].to_vec());
            }
            return None;
        }
        visited.insert(loc, true);
        path.push(loc);
        for loc2 in net[loc].outputs.keys() {
            if let Some(c) = dfs(net, *loc2, visited, path) {
                return Some(c);
            }
        }
        path.pop();
        visited.insert(loc, false);
        None
    }
    let mut visited = HM::new();
    for i in 0..I {
        if let Some(c) = dfs(net, NeuronLocation::Input(i), &mut visited, &mut vec![]) {
            return Some(c);
        }
    }
    for i in 0..net.hidden_layers.len() {
        let loc = NeuronLocation::Hidden(i);
        if !visited.contains_key(&loc) {
            if let Some(c) = dfs(net, loc, &mut visited, &mut vec![]) {
                return Some(c);
            }
        }
    }
    None
}

#[test]
fn debug_locate_cycle_source() {
    // Run with no mutations to see if remove_cycles itself fails
    let mut settings_no_mut = ReproductionSettings::default();
    settings_no_mut.mutation_passes = 0;

    let mut found_no_mut = false;
    let mut found_with_mut = false;

    'outer: for seed in 0..300u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net1 = NeuralNetwork::<10, 10>::new(&mut rng);
        let mut net2 = NeuralNetwork::<10, 10>::new(&mut rng);
        for iter in 0..100usize {
            let a = net1.crossover(&net2, &settings_no_mut, 0.25, &mut rng);
            let b = net2.crossover(&net1, &settings_no_mut, 0.25, &mut rng);
            if let Some(cycle) = find_cycle_helper(&a) {
                println!("remove_cycles FAILED: seed={} iter={} (a): {:?}", seed, iter, cycle);
                found_no_mut = true;
                break 'outer;
            }
            if let Some(cycle) = find_cycle_helper(&b) {
                println!("remove_cycles FAILED: seed={} iter={} (b): {:?}", seed, iter, cycle);
                found_no_mut = true;
                break 'outer;
            }
            net1 = a;
            net2 = b;
        }
    }
    if !found_no_mut {
        println!("remove_cycles seems correct (no cycles in 300 seeds x 100 iters without mutation)");
    }

    // Run with mutations to see if mutation introduces cycles
    let settings_with_mut = ReproductionSettings::default(); // mutation_passes = 3

    'outer2: for seed in 0..300u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net1 = NeuralNetwork::<10, 10>::new(&mut rng);
        let mut net2 = NeuralNetwork::<10, 10>::new(&mut rng);
        for iter in 0..50usize {
            let a = net1.crossover(&net2, &settings_with_mut, 0.25, &mut rng);
            let b = net2.crossover(&net1, &settings_with_mut, 0.25, &mut rng);
            if let Some(cycle) = find_cycle_helper(&a) {
                println!("Mutation introduced cycle: seed={} iter={} (a): {:?}", seed, iter, cycle);
                found_with_mut = true;
                break 'outer2;
            }
            if let Some(cycle) = find_cycle_helper(&b) {
                println!("Mutation introduced cycle: seed={} iter={} (b): {:?}", seed, iter, cycle);
                found_with_mut = true;
                break 'outer2;
            }
            net1 = a;
            net2 = b;
        }
    }
    if !found_with_mut {
        println!("Mutations don't introduce cycles either (no cycles found)");
    }
    
    assert!(!found_no_mut, "remove_cycles is broken");
    assert!(!found_with_mut, "mutation is adding cycles (is_connection_safe is broken)");
}

#[test]
fn debug_find_bad_connection() {
    // Reproduce: seed=0, iter=47 introduces a cycle
    let mut rng = StdRng::seed_from_u64(0);
    let mut net1 = NeuralNetwork::<10, 10>::new(&mut rng);
    let mut net2 = NeuralNetwork::<10, 10>::new(&mut rng);
    let settings = ReproductionSettings::default();

    for iter in 0..47usize {
        let a = net1.crossover(&net2, &settings, 0.25, &mut rng);
        let b = net2.crossover(&net1, &settings, 0.25, &mut rng);
        net1 = a;
        net2 = b;
    }

    // Now do crossover 47 (iter=47) step by step
    // net1.crossover(&net2...) produces 'a'
    // Try with mutation_passes=0 to see if problem is in merge or mutations
    let mut settings_0 = settings.clone();
    settings_0.mutation_passes = 0;
    let a0 = net1.crossover(&net2, &settings_0, 0.25, &mut StdRng::seed_from_u64(47_000));
    let cyc0 = find_cycle_helper(&a0);
    println!("iter=47, mutation_passes=0 cycle: {:?}", cyc0);
    
    let mut settings_1 = settings.clone();
    settings_1.mutation_passes = 1;
    let a1 = net1.crossover(&net2, &settings_1, 0.25, &mut StdRng::seed_from_u64(47_000));
    let cyc1 = find_cycle_helper(&a1);
    println!("iter=47, mutation_passes=1 cycle: {:?}", cyc1);
    
    let mut settings_2 = settings.clone();
    settings_2.mutation_passes = 2;
    let a2 = net1.crossover(&net2, &settings_2, 0.25, &mut StdRng::seed_from_u64(47_000));
    let cyc2 = find_cycle_helper(&a2);
    println!("iter=47, mutation_passes=2 cycle: {:?}", cyc2);
    
    let a3 = net1.crossover(&net2, &settings, 0.25, &mut StdRng::seed_from_u64(47_000));
    let cyc3 = find_cycle_helper(&a3);
    println!("iter=47, mutation_passes=3 cycle: {:?}", cyc3);
    
    // Also check net2.crossover(net1)
    let b0 = net2.crossover(&net1, &settings_0, 0.25, &mut StdRng::seed_from_u64(47_001));
    println!("iter=47 b, mutation_passes=0 cycle: {:?}", find_cycle_helper(&b0));
    let b3 = net2.crossover(&net1, &settings, 0.25, &mut StdRng::seed_from_u64(47_001));
    println!("iter=47 b, mutation_passes=3 cycle: {:?}", find_cycle_helper(&b3));
}

#[test]
fn debug_find_bad_connection2() {
    // Reproduce: seed=0, iter=47 introduces a cycle
    // Must use the SAME rng throughout
    let mut rng = StdRng::seed_from_u64(0);
    let mut net1 = NeuralNetwork::<10, 10>::new(&mut rng);
    let mut net2 = NeuralNetwork::<10, 10>::new(&mut rng);
    let settings = ReproductionSettings::default();

    for iter in 0..47usize {
        let a = net1.crossover(&net2, &settings, 0.25, &mut rng);
        let b = net2.crossover(&net1, &settings, 0.25, &mut rng);
        net1 = a;
        net2 = b;
    }

    println!("net1 hidden len: {}", net1.hidden_layers.len());
    println!("net2 hidden len: {}", net2.hidden_layers.len());
    
    // Now at iteration 47, the actual test does:
    // a = net1.crossover(&net2, ...)
    // b = net2.crossover(&net1, ...)
    // And the cycle shows up in 'a'
    
    // Test 'a' with 0, 1, 2, 3 mutation passes, using the current rng state
    let settings_3 = settings.clone();
    
    // We need separate snapshots of rng state for each test
    // But since we can't clone StdRng, let's just do it sequentially
    
    // Do 0 mutations
    let mut s0 = settings.clone();
    s0.mutation_passes = 0;
    // Can't replay rng here... let's just do the actual crossover and check
    
    // Let's just do the full mutation=3 crossover and check step-by-step
    // by doing the crossover merge first (mutation_passes=0)
    // and checking after each mutation pass
    let a = net1.crossover(&net2, &settings_3, 0.25, &mut rng);
    let cycle = find_cycle_helper(&a);
    println!("a (full crossover) cycle: {:?}", cycle);
    
    let b = net2.crossover(&net1, &settings_3, 0.25, &mut rng);
    let cycle_b = find_cycle_helper(&b);
    println!("b (full crossover) cycle: {:?}", cycle_b);
}

#[test]
fn debug_add_connection_cycle() {
    // Try to find a case where add_connection adds a cyclic connection
    let settings_with_mut = ReproductionSettings::default();
    
    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net1 = NeuralNetwork::<10, 10>::new(&mut rng);
        let mut net2 = NeuralNetwork::<10, 10>::new(&mut rng);
        
        for iter in 0..50usize {
            let a = net1.crossover(&net2, &settings_with_mut, 0.25, &mut rng);
            let b = net2.crossover(&net1, &settings_with_mut, 0.25, &mut rng);
            
            // Double-check: is_connection_safe should return false for any existing cycle
            for i in 0..10usize {
                let from = NeuronLocation::Input(i);
                let n = &a[from];
                for &to in n.outputs.keys() {
                    // Check if is_connection_safe correctly returns false for reverse connection
                    if let NeuronLocation::Hidden(_) | NeuronLocation::Output(_) = to {
                        // don't test
                    }
                }
            }
            
            let cycle_a = find_cycle_helper(&a);
            let cycle_b = find_cycle_helper(&b);
            
            if cycle_a.is_some() || cycle_b.is_some() {
                println!("seed={} iter={} cycle_a={:?} cycle_b={:?}", seed, iter, cycle_a, cycle_b);
                // Print the first cycle node's connections
                if let Some(ref cycle) = cycle_a {
                    for &node in cycle {
                        println!("  {:?} -> {:?}", node, a[node].outputs.keys().collect::<Vec<_>>());
                    }
                }
                // Check if is_connection_safe would detect the cycle
                if let Some(ref cycle) = cycle_a {
                    let n = cycle.len();
                    for i in 0..n {
                        let from = cycle[i];
                        let to = cycle[(i + 1) % n];
                        // The edge from -> to creates a cycle, so is_connection_safe should return false
                        // But does it?
                        let safe = a.is_connection_safe(Connection { from, to });
                        println!("  is_connection_safe({:?} -> {:?}) = {}", from, to, safe);
                        // If safe returns true, that means the check is broken
                        // (this connection already exists, but it should also detect the EXISTING cycle)
                    }
                }
                return;
            }
            
            net1 = a;
            net2 = b;
        }
    }
    println!("No cycles found!");
}
