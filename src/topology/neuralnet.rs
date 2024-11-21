use genetic_rs::prelude::*;
use rand::Rng;

use crate::activation_fn;
use super::activation::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MutationSettings {
    pub mutation_rate: f32,
    pub mutation_passes: f32,
}

impl Default for MutationSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.01,
            mutation_passes: 3.,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkTopology<const I: usize, const O: usize> {
    pub input_layer: [NeuronTopology; I],
    pub hidden_layers: Vec<NeuronTopology>,
    pub output_layer: [NeuronTopology; O],
    pub mutation_settings: MutationSettings,
}

impl<const I: usize, const O: usize> NeuralNetworkTopology<I, O> {
    pub fn new(mutation_settings: MutationSettings, rng: &mut impl Rng) -> Self {
        let mut output_layer = Vec::with_capacity(O);

        for _ in 0..O {
            output_layer.push(NeuronTopology::new_with_activation(
                vec![],
                activation_fn!(sigmoid),
                rng,
            ));
        }

        let output_layer = output_layer.try_into().unwrap();

        let mut input_layer = Vec::with_capacity(I);

        for _ in 0..I {
            let outputs = (0..rng.gen_range(1..=O))
                .map(|_| {
                    let mut already_chosen = Vec::new();
                    let mut i = rng.gen_range(0..O);
                    while already_chosen.contains(&i) {
                        i = rng.gen_range(0..O);
                    }

                    already_chosen.push(i);

                    (NeuronLocation::Output(i), rng.gen())
                })
                .collect();

            input_layer.push(NeuronTopology::new_with_activation(outputs, activation_fn!(sigmoid), rng));
        }

        let input_layer = input_layer.try_into().unwrap();

        Self {
            input_layer,
            hidden_layers: vec![],
            output_layer,
            mutation_settings,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuronTopology {
    pub outputs: Vec<(NeuronLocation, f32)>,
    pub bias: f32,
    pub activation_fn: ActivationFn,
}

impl NeuronTopology {
    pub fn new_with_activation(outputs: Vec<(NeuronLocation, f32)>, activation_fn: ActivationFn, rng: &mut impl Rng) -> Self {
        Self {
            outputs,
            bias: rng.gen(),
            activation_fn,
        }
    }
}

/// A pseudo-pointer of sorts used to make structural conversions very fast and easy to write.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NeuronLocation {
    /// Points to a neuron in the input layer at contained index.
    Input(usize),

    /// Points to a neuron in the hidden layer at contained index.
    Hidden(usize),

    /// Points to a neuron in the output layer at contained index.
    Output(usize),
}

impl NeuronLocation {
    /// Returns `true` if it points to the input layer. Otherwise, returns `false`.
    pub fn is_input(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    /// Returns `true` if it points to the hidden layer. Otherwise, returns `false`.
    pub fn is_hidden(&self) -> bool {
        matches!(self, Self::Hidden(_))
    }

    /// Returns `true` if it points to the output layer. Otherwise, returns `false`.
    pub fn is_output(&self) -> bool {
        matches!(self, Self::Output(_))
    }

    /// Retrieves the index value, regardless of layer. Does not consume.
    pub fn unwrap(&self) -> usize {
        match self {
            Self::Input(i) => *i,
            Self::Hidden(i) => *i,
            Self::Output(i) => *i,
        }
    }
}