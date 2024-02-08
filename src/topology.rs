use std::sync::{Arc, RwLock};

use genetic_rs::prelude::*;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct NeuralNetworkTopology {
    pub input_layer: Vec<Arc<RwLock<NeuronTopology>>>,
    pub hidden_layers: Vec<Arc<RwLock<NeuronTopology>>>,
    pub output_layer: Vec<Arc<RwLock<NeuronTopology>>>,
    pub mutation_rate: f32,
}

impl NeuralNetworkTopology {
    pub fn new(inputs: usize, outputs: usize, mutation_rate: f32, rng: &mut impl Rng) -> Self {
        let mut input_layer = Vec::with_capacity(inputs);

        for _ in 0..inputs {
            input_layer.push(Arc::new(RwLock::new(NeuronTopology::new(vec![], rng))));
        }

        let mut output_layer = Vec::with_capacity(outputs);
        let input_locs: Vec<_> = input_layer
            .iter()
            .enumerate()
            .map(|(i, _n)| NeuronLocation::Input(i))
            .collect();

        for _ in 0..outputs {
            let mut already_chosen = Vec::new();

            // random number of connections to random input neurons.
            let input = (0..rng.gen_range(0..inputs))
                .map(|_| {
                    let mut i = rng.gen_range(0..inputs);
                    while already_chosen.contains(&i) {
                        i = rng.gen_range(0..inputs);
                    }

                    already_chosen.push(i);

                    input_locs[i]
                })
                .collect();

            output_layer.push(Arc::new(RwLock::new(NeuronTopology::new(input, rng))));
        }

        Self {
            input_layer,
            hidden_layers: vec![],
            output_layer,
            mutation_rate,
        }
    }

    fn is_connection_cyclic(&self, loc1: NeuronLocation, loc2: NeuronLocation) -> bool {
        if loc1 == loc2 {
            return true;
        }

        for &(n, _w) in &self.get_neuron(loc2).read().unwrap().inputs {
            if self.is_connection_cyclic(n, loc2) {
                return true;
            }
        }

        false
    }

    pub fn get_neuron(&self, loc: NeuronLocation) -> Arc<RwLock<NeuronTopology>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    pub fn rand_neuron(&self, rng: &mut impl Rng) -> (Arc<RwLock<NeuronTopology>>, NeuronLocation) {
        match rng.gen_range(0..3) {
            0 => {
                let i = rng.gen_range(0..self.input_layer.len());
                (self.input_layer[i].clone(), NeuronLocation::Input(i))
            },
            1 => {
                let i = rng.gen_range(0..self.hidden_layers.len());
                (self.hidden_layers[i].clone(), NeuronLocation::Hidden(i))
            },
            _ => {
                let i = rng.gen_range(0..self.output_layer.len());
                (self.output_layer[i].clone(), NeuronLocation::Output(i))
            }
        }
    }
}

impl RandomlyMutable for NeuralNetworkTopology {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
       if rng.gen::<f32>() <= rate { 
            // split preexisting connection
            let (mut n2, _) = self.rand_neuron(rng);
                    
            while n2.read().unwrap().inputs.len() == 0 {
                (n2, _) = self.rand_neuron(rng);
            }

            let mut n2: std::sync::RwLockWriteGuard<'_, NeuronTopology> = n2.write().unwrap();
            let i = rng.gen_range(0..n2.inputs.len());
            let (loc, w) = n2.inputs.remove(i);

            let loc3 = NeuronLocation::Hidden(self.hidden_layers.len());
            self.hidden_layers.push(Arc::new(RwLock::new(NeuronTopology::new(vec![loc], rng))));

            n2.inputs.insert(i, (loc3, w));
       }

        if rng.gen::<f32>() <= rate {
            // add a connection
            let (mut n1, mut loc1) = self.rand_neuron(rng);

            while n1.read().unwrap().inputs.len() == 0 {
                (n1, loc1) = self.rand_neuron(rng);
            }

            let (mut n2, mut loc2) = self.rand_neuron(rng);

            while self.is_connection_cyclic(loc1, loc2) {
                (n2, loc2) = self.rand_neuron(rng);
            }

            n2.write().unwrap().inputs.push((loc1, rng.gen()));
        }

        if rng.gen::<f32>() <= rate {
            // mutate a connection
            let (mut n, _) = self.rand_neuron(rng);

            while n.read().unwrap().inputs.len() == 0 {
                (n, _) = self.rand_neuron(rng);
            }

            let mut n = n.write().unwrap();
            let i = rng.gen_range(0..n.inputs.len());
            let (_, w) = &mut n.inputs[i];
            *w += rng.gen::<f32>() * rate; 
       }
    }
}

impl DivisionReproduction for NeuralNetworkTopology {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(self.mutation_rate, rng);
        child        
    }
}

#[cfg(feature = "crossover")]
impl CrossoverReproduction for NeuralNetworkTopology {
    fn spawn_child(&self, other: &Self, rng: &mut impl Rng) -> Self {
        todo!();
    }
}

impl Prunable for NeuralNetworkTopology {}

#[derive(Debug, Clone)]
pub struct NeuronTopology {
    pub inputs: Vec<(NeuronLocation, f32)>,
    pub bias: f32,
}

impl NeuronTopology {
    pub fn new(inputs: Vec<NeuronLocation>, rng: &mut impl Rng) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|i| (i, rng.gen::<f32>()))
            .collect();

        Self {
            inputs,
            bias: rng.gen(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuronLocation {
    Input(usize),
    Hidden(usize),
    Output(usize),
}

impl NeuronLocation {
    pub fn is_input(&self) -> bool {
        match self {
            Self::Input(_) => true,
            _ => false,
        }
    }

    pub fn is_hidden(&self) -> bool {
        match self {
            Self::Hidden(_) => true,
            _ => false,
        }
    }

    pub fn is_output(&self) -> bool {
        match self {
            Self::Output(_) => true,
            _ => false,
        }
    }

    pub fn unwrap(&self) -> usize {
        match self {
            Self::Input(i) => *i,
            Self::Hidden(i) => *i,
            Self::Output(i) => *i,
        }
    }
}