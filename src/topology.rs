use genetic_rs::prelude::*;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct NeuralNetworkTopology {
    pub input_layer: Vec<NeuronTopology>,
    pub hidden_layer: Vec<NeuronTopology>,
    pub output_layer: Vec<NeuronTopology>,
    pub mutation_rate: f32,
}

impl NeuralNetworkTopology {
    pub fn new(inputs: usize, outputs: usize, mutation_rate: f32, rng: &mut impl Rng) -> Self {
        let mut input_layer = Vec::with_capacity(inputs);

        for _ in 0..inputs {
            input_layer.push(NeuronTopology::new(vec![], rng));
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

            output_layer.push(NeuronTopology::new(input, rng));
        }

        Self {
            input_layer,
            hidden_layer: vec![],
            output_layer,
            mutation_rate,
        }
    }
}

impl RandomlyMutable for NeuralNetworkTopology {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        todo!();
    }
}

#[cfg(not(feature = "crossover"))]
impl DivisionReproduction for NeuralNetworkTopology {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(self.mutation_rate, rng);
        child        
    }
}

impl Prunable for NeuralNetworkTopology {}

#[derive(Debug, Clone)]
pub struct NeuronTopology {
    inputs: Vec<(NeuronLocation, f32)>,
    bias: f32,
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