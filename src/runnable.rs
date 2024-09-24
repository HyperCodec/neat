use crate::topology::*;

#[cfg(not(feature = "rayon"))]
use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use std::sync::{Arc, RwLock};

/// A runnable, stated Neural Network generated from a [NeuralNetworkTopology]. Use [`NeuralNetwork::from`] to go from stateles to runnable.
#[derive(Debug)]
pub struct NeuralNetwork<'a, const I: usize, const O: usize> {
    topology: &'a NeuralNetworkTopology<I, O>,
}

impl<const I: usize, const O: usize> NeuralNetwork<'_, I, O> {
    /// Predicts an output for the given inputs.
    #[cfg(not(feature = "rayon"))]
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        let mut state_cache = HashMap::new();

        for (i, v) in inputs.iter().enumerate() {
            state_cache.insert(NeuronLocation::Input(i), *v);
        }

        (0..O)
            .map(NeuronLocation::Output)
            .map(|loc| self.process_neuron(loc, &mut state_cache))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Parallelized prediction of outputs from inputs.
    #[cfg(feature = "rayon")]
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        inputs.par_iter().enumerate().for_each(|(i, v)| {
            let mut nw = self.input_layer[i].write().unwrap();
            nw.state.value = *v;
            nw.state.processed = true;
        });

        (0..O)
            .map(NeuronLocation::Output)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|loc| self.process_neuron(loc))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[cfg(not(feature = "rayon"))]
    fn process_neuron(&self, loc: NeuronLocation, cache: &mut HashMap<NeuronLocation, f32>) -> f32 {
        if let Some(v) = cache.get(&loc) {
            return *v;
        }

        let n = self.get_neuron(loc).unwrap();
        let mut v = 0.;
        for (l, w) in &n.inputs {
            v += self.process_neuron(*l, cache) * w;
        }

        v = n.activate(v);

        cache.insert(loc, v);

        v
    }

    #[cfg(feature = "rayon")]
    fn process_neuron(&self, loc: NeuronLocation) -> f32 {
        let n = self.get_neuron(loc);

        {
            let nr = n.read().unwrap();

            if nr.state.processed {
                return nr.state.value;
            }
        }

        let val: f32 = n
            .read()
            .unwrap()
            .inputs
            .par_iter()
            .map(|&(n2, w)| {
                let processed = self.process_neuron(n2);
                processed * w
            })
            .sum();

        let mut nw = n.write().unwrap();
        nw.state.value += val;
        nw.activate();

        nw.state.value
    }

    #[cfg(not(feature = "rayon"))]
    fn get_neuron<'a>(&self, loc: NeuronLocation) -> Option<&'a NeuronTopology> {
        match loc {
            NeuronLocation::Input(i) => self.topology.input_layer.get(i),
            NeuronLocation::Hidden(i) => self.topology.hidden_layers.get(i),
            NeuronLocation::Output(i) => self.topology.output_layer.get(i),
        }
    }

    #[cfg(feature = "rayon")]
    fn get_neuron(&self, loc: NeuronLocation) -> Arc<RwLock<Neuron>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    /// Flushes the neural network's state.
    #[cfg(feature = "rayon")]
    pub fn flush_state(&self) {
        self.input_layer
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());

        self.hidden_layers
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());

        self.output_layer
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());
    }
}

impl<'a, const I: usize, const O: usize> From<&'a NeuralNetworkTopology<I, O>>
    for NeuralNetwork<'a, I, O>
{
    #[cfg(not(feature = "rayon"))]
    fn from(topology: &'a NeuralNetworkTopology<I, O>) -> Self {
        Self { topology }
    }

    #[cfg(feature = "rayon")]
    fn from(value: &NeuralNetworkTopology<I, O>) -> Self {
        let input_layer = value
            .input_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value
            .hidden_layers
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

        let output_layer = value
            .output_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}

/// A blanket trait for iterators meant to help with interpreting the output of a [`NeuralNetwork`]
#[cfg(feature = "max-index")]
pub trait MaxIndex<T: PartialOrd> {
    /// Retrieves the index of the max value.
    fn max_index(self) -> usize;
}

#[cfg(feature = "max-index")]
impl<I: Iterator<Item = T>, T: PartialOrd> MaxIndex<T> for I {
    // slow and lazy implementation but it works (will prob optimize in the future)
    fn max_index(self) -> usize {
        self.enumerate()
            .max_by(|(_, v), (_, v2)| v.partial_cmp(v2).unwrap())
            .unwrap()
            .0
    }
}
