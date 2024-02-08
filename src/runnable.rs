use crate::topology::*;
use std::sync::{Arc, RwLock};

#[cfg(feature = "rayon")] use rayon::prelude::*;

pub struct NeuralNetwork<const I: usize, const O: usize> {
    input_layer: [Arc<RwLock<Neuron>>; I],
    hidden_layers: Vec<Arc<RwLock<Neuron>>>,
    output_layer: [Arc<RwLock<Neuron>>; O],
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    #[cfg(not(feature = "rayon"))]
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        for (i, v) in inputs.iter().enumerate() {
            self.input_layer[i].write().unwrap().state.value = *v;
        }

        (0..O)
            .map(|i| NeuronLocation::Output(i))
            .map(|loc| self.process_neuron(loc))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[cfg(feature = "rayon")]
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        inputs.par_iter().enumerate().for_each(|(i, v)| {
            self.input_layer[i].write().unwrap().state.value = *v;
        });

        (0..O)
            .into_par_iter()
            .map(|i| NeuronLocation::Output(i))
            .map(|loc| self.process_neuron(loc))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn get_neuron(&self, loc: NeuronLocation) -> Arc<RwLock<Neuron>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    #[cfg(not(feature = "rayon"))]
    pub fn flush_state(&self) {
        for n in &self.input_layer {
            n.write().unwrap().flush_state();
        }

        for n in &self.hidden_layers {
            n.write().unwrap().flush_state();
        }
        
        for n in &self.output_layer {
            n.write().unwrap().flush_state();
        }
    }
    
    #[cfg(feature = "rayon")]
    pub fn flush_state(&self) {
        self.input_layer.par_iter().for_each(|n| n.write().unwrap().flush_state());

        self.hidden_layers.par_iter().for_each(|n| n.write().unwrap().flush_state());
        
        self.output_layer.par_iter().for_each(|n| n.write().unwrap().flush_state());
    }

    #[cfg(not(feature = "rayon"))]
    pub fn process_neuron(&self, loc: NeuronLocation) -> f32 {
        let n = self.get_neuron(loc);

        {
            let nr = n.read().unwrap();

            if nr.state.processed {
                return nr.state.value;
            }
        }

        let mut n = n.try_write().unwrap();

        for (l, w) in n.inputs.clone() {
            n.state.value += self.process_neuron(l) * w;
        }

        n.write().unwrap().sigmoid();

        n.state.value
    }

    #[cfg(feature = "rayon")]
    pub fn process_neuron(&self, loc: NeuronLocation) -> f32 {
        let n = self.get_neuron(loc);

        {
            let nr = n.read().unwrap();

            if nr.state.processed {
                return nr.state.value;
            }
        }

        n.read().unwrap().inputs
            .clone()
            .into_par_iter()
            .for_each(|(n2, w)| {
                let processed = self.process_neuron(n2); // separate step so write lock doesnt block process_neuron on other threads
                n.write().unwrap().state.value += processed * w
            });

        n.write().unwrap().sigmoid();

        let nr = n.read().unwrap();
        nr.state.value
    }

}

impl<const I: usize, const O: usize> From<&NeuralNetworkTopology<I, O>> for NeuralNetwork<I, O> {
    fn from(value: &NeuralNetworkTopology<I, O>) -> Self {
        let input_layer = value.input_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value.hidden_layers
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

        let output_layer = value.output_layer
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

#[derive(Clone, Debug)]
pub struct Neuron {
    inputs: Vec<(NeuronLocation, f32)>,
    bias: f32,
    state: NeuronState,
}

impl Neuron {
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
    }

    pub fn sigmoid(&mut self) {
        self.state.value = 1. / (1. + std::f32::consts::E.powf(-self.state.value))
    }
}

impl From<&NeuronTopology> for Neuron {
    fn from(value: &NeuronTopology) -> Self {
        Self {
            inputs: value.inputs.clone(),
            bias: value.bias,
            state: NeuronState {
                value: value.bias,
                ..Default::default()
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NeuronState {
    pub value: f32,
    pub processed: bool,
}

pub trait MaxIndex<T: PartialOrd> {
    fn max_index(self) -> usize;
}

impl<I: Iterator<Item = T>, T: PartialOrd> MaxIndex<T> for I {
    // slow and lazy implementation but it works (will prob optimize in the future)
    fn max_index(self) -> usize {
        self
            .enumerate()
            .max_by(|(_, v), (_, v2)| v.partial_cmp(v2).unwrap())
            .unwrap()
            .0
    }
}