use crate::*;
use genetic_rs::prelude::rand::{SeedableRng, rngs::StdRng};
use union_find::{QuickFindUf, UnionBySize, UnionFind};

fn loc_to_index<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>, loc: NeuronLocation) -> usize {
    match loc {
        NeuronLocation::Input(i) => i,
        NeuronLocation::Hidden(i) => I + i,
        NeuronLocation::Output(i) => I + net.hidden_layers.len() + i,
    }
}

fn assert_graph_invariants<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) {
    let total_len = I + O + net.hidden_layers.len();
    let mut uf = QuickFindUf::<UnionBySize>::new(total_len);
    
    for i in 0..I {
        let loc = NeuronLocation::Input(i);
        let a_ident = uf.find(i);

        let n = net.get_neuron(loc);
        for (loc2, _) in &n.outputs {
            let b_ident = uf.find(loc_to_index(net, *loc2));
            if !uf.union(a_ident, b_ident) {
                panic!("cycle detected in network: {loc:?} -> {loc2:?}");
            }
        }
    }

    for i in 0..total_len {
        if uf.find(i) >= I {
            panic!("found hanging neuron");
        }
    }
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
        let n = net.get_neuron(NeuronLocation::Input(i));
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
            // maybe redundant because of graph invariants, but oh well
            panic!("found hanging neuron");
        }

        assert_eq!(x, net.hidden_layers[i].input_count);
    }

    for (i, x) in cache.output.into_iter().enumerate() {
        assert_eq!(x, net.output_layer[i].input_count);
    }
}

fn assert_network_invariants<const I: usize, const O: usize>(net: &NeuralNetwork<I, O>) {
    // assert_graph_invariants(net);
    assert_cache_consistency(net);
    // TODO other invariants
}

const TEST_COUNT: u64 = 1000;
fn rng_test(test: impl Fn(&mut StdRng)) {
    for seed in 0..TEST_COUNT {
        let mut rng = StdRng::seed_from_u64(seed);
        test(&mut rng);
    }
}

#[test]
fn create_network() {
    rng_test(|rng| {
        let net = NeuralNetwork::<10, 10>::new(MutationSettings::default(), rng);
        assert_network_invariants(&net);
    });
}

#[test]
fn split_connection() {
    // rng doesn't matter here since it's just adding bias in eval
    let mut rng = StdRng::seed_from_u64(0xabcdef);

    let mut net = NeuralNetwork::<1, 1>::new(MutationSettings::default(), &mut rng);
    assert_network_invariants(&net);
    
    net.split_connection(Connection { from: NeuronLocation::Input(0), to: NeuronLocation::Output(0) }, &mut rng);
    assert_network_invariants(&net);

    assert_eq!(*net.input_layer[0].outputs.keys().next().unwrap(), NeuronLocation::Hidden(0));
    assert_eq!(*net.hidden_layers[0].outputs.keys().next().unwrap(), NeuronLocation::Output(0));
}

const NUM_MUTATIONS: usize = 1000;
const MUTATION_RATE: f32 = 0.25;
#[test]
fn mutate() {
    rng_test(|rng| {
        let mut net = NeuralNetwork::<10, 10>::new(MutationSettings::default(), rng);
        assert_network_invariants(&net);

        for _ in 0..NUM_MUTATIONS {
            net.mutate(MUTATION_RATE, rng);
            assert_network_invariants(&net);
        }
    });
}