use crate::topology::*;
use std::sync::{Arc, RwLock};

pub struct NeuralNetwork {
    input_layer: Vec<Arc<RwLock<Neuron>>>,
    hidden_layers: Vec<Arc<RwLock<Neuron>>>,
    output_layer: Vec<Arc<RwLock<Neuron>>>,
}

impl NeuralNetwork {
    pub fn predict(&mut self, inputs: Vec<f32>) -> Vec<f32> {
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

    pub fn get_neuron(&self, loc: NeuronLocation) -> Arc<RwLock<Neuron>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    pub fn flush_state(&mut self) {
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

    fn process_neuron(&mut self, loc: NeuronLocation) -> f32 {
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

        n.state.value
    }

}

impl From<&NeuralNetworkTopology> for NeuralNetwork {
    fn from(value: &NeuralNetworkTopology) -> Self {
        let input_layer = value.input_layer
            .iter()
            .map(Neuron::from)
            .map(RwLock::from)
            .map(Arc::from)
            .collect();

        let hidden_layers = value.hidden_layers
            .iter()
            .map(Neuron::from)
            .map(RwLock::from)
            .map(Arc::from)
            .collect();

        let output_layer = value.output_layer
            .iter()
            .map(Neuron::from)
            .map(RwLock::from)
            .map(Arc::from)
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