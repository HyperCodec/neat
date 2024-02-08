use crate::topology::*;
use std::sync::{Arc, RwLock};

#[cfg(feature = "rayon")] use rayon::prelude::*;

pub struct NeuralNetwork {
    input_layer: Vec<Arc<RwLock<Neuron>>>,
    hidden_layers: Vec<Arc<RwLock<Neuron>>>,
    output_layer: Vec<Arc<RwLock<Neuron>>>,
}

impl NeuralNetwork {
    #[cfg(not(feature = "rayon"))]
    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        if self.input_layer.len() != inputs.len() {
            panic!("Invalid input layer specified. Expected {}, got {}", self.input_layer.len(), inputs.len());
        }

        for (i, v) in inputs.iter().enumerate() {
            self.input_layer[i].write().unwrap().state.value = *v;
        }

        (0..self.output_layer.len())
            .map(|i| NeuronLocation::Output(i))
            .map(|loc| self.process_neuron(loc))
            .collect()
    }

    #[cfg(feature = "rayon")]
    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        if self.input_layer.len() != inputs.len() {
            panic!("Invalid input layer specified. Expected {}, got {}", self.input_layer.len(), inputs.len());
        }

        inputs.par_iter().enumerate().for_each(|(i, v)| {
            self.input_layer[i].write().unwrap().state.value = *v;
        });

        (0..self.output_layer.len())
            .into_par_iter()
            .map(|i| NeuronLocation::Output(i))
            .map(|loc| self.process_neuron(loc))
            .collect()
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

impl From<&NeuralNetworkTopology> for NeuralNetwork {
    fn from(value: &NeuralNetworkTopology) -> Self {
        let input_layer = value.input_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

        let hidden_layers = value.hidden_layers
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

        let output_layer = value.output_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

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