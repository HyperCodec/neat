/// Contains useful structs for serializing/deserializing a [`NeuronTopology`]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
#[cfg(feature = "serde")]
pub mod nnt_serde;

/// Contains structs and traits used for activation functions.
pub mod activation;

pub use activation::*;

use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};

use genetic_rs::prelude::*;
use rand::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::activation_fn;

/// A stateless neural network topology.
/// This is the struct you want to use in your agent's inheritance.
/// See [`NeuralNetwork::from`][crate::NeuralNetwork::from] for how to convert this to a runnable neural network.
#[derive(Debug)]
pub struct NeuralNetworkTopology<const I: usize, const O: usize> {
    /// The input layer of the neural network. Uses a fixed length of `I`.
    pub input_layer: [Arc<RwLock<NeuronTopology>>; I],

    /// The hidden layers of the neural network. Because neurons have a flexible connection system, all of them exist in the same flat vector.
    pub hidden_layers: Vec<Arc<RwLock<NeuronTopology>>>,

    /// The output layer of the neural netowrk. Uses a fixed length of `O`.
    pub output_layer: [Arc<RwLock<NeuronTopology>>; O],

    /// The mutation rate used in [`NeuralNetworkTopology::mutate`] after crossover/division.
    pub mutation_rate: f32,

    /// The number of mutation passes (and thus, maximum number of possible mutations that can occur for each entity in the generation).
    pub mutation_passes: usize,
}

impl<const I: usize, const O: usize> NeuralNetworkTopology<I, O> {
    /// Creates a new [`NeuralNetworkTopology`].
    pub fn new(mutation_rate: f32, mutation_passes: usize, rng: &mut impl Rng) -> Self {
        let input_layer: [Arc<RwLock<NeuronTopology>>; I] = (0..I)
            .map(|_| {
                Arc::new(RwLock::new(NeuronTopology::new_with_activation(
                    vec![],
                    activation_fn!(linear_activation),
                    rng,
                )))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut output_layer = Vec::with_capacity(O);

        for _ in 0..O {
            // random number of connections to random input neurons.
            let input = (0..rng.gen_range(1..=I))
                .map(|_| {
                    let mut already_chosen = Vec::new();
                    let mut i = rng.gen_range(0..I);
                    while already_chosen.contains(&i) {
                        i = rng.gen_range(0..I);
                    }

                    already_chosen.push(i);

                    NeuronLocation::Input(i)
                })
                .collect();

            output_layer.push(Arc::new(RwLock::new(NeuronTopology::new_with_activation(
                input,
                activation_fn!(sigmoid),
                rng,
            ))));
        }

        let output_layer = output_layer.try_into().unwrap();

        Self {
            input_layer,
            hidden_layers: vec![],
            output_layer,
            mutation_rate,
            mutation_passes,
        }
    }

    /// Creates a new connection between the neurons.
    /// If the connection is cyclic, it does not add a connection and returns false.
    /// Otherwise, it returns true.
    pub fn add_connection(
        &mut self,
        from: NeuronLocation,
        to: NeuronLocation,
        weight: f32,
    ) -> bool {
        if self.is_connection_cyclic(from, to) {
            return false;
        }

        // Add the connection since it is not cyclic
        self.get_neuron(to)
            .write()
            .unwrap()
            .inputs
            .push((from, weight));

        true
    }

    fn is_connection_cyclic(&self, from: NeuronLocation, to: NeuronLocation) -> bool {
        if to.is_input() || from.is_output() {
            return true;
        }

        let mut visited = HashSet::new();
        self.dfs(from, to, &mut visited)
    }

    // TODO rayon implementation
    fn dfs(
        &self,
        current: NeuronLocation,
        target: NeuronLocation,
        visited: &mut HashSet<NeuronLocation>,
    ) -> bool {
        if current == target {
            return true;
        }

        visited.insert(current);

        let n = self.get_neuron(current);
        let nr = n.read().unwrap();

        for &(input, _) in &nr.inputs {
            if !visited.contains(&input) && self.dfs(input, target, visited) {
                return true;
            }
        }

        visited.remove(&current);
        false
    }

    /// Gets a neuron pointer from a [`NeuronLocation`].
    /// You shouldn't ever need to directly call this unless you are doing complex custom mutations.
    pub fn get_neuron(&self, loc: NeuronLocation) -> Arc<RwLock<NeuronTopology>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    /// Gets a random neuron and its location.
    pub fn rand_neuron(&self, rng: &mut impl Rng) -> (Arc<RwLock<NeuronTopology>>, NeuronLocation) {
        match rng.gen_range(0..3) {
            0 => {
                let i = rng.gen_range(0..self.input_layer.len());
                (self.input_layer[i].clone(), NeuronLocation::Input(i))
            }
            1 if !self.hidden_layers.is_empty() => {
                let i = rng.gen_range(0..self.hidden_layers.len());
                (self.hidden_layers[i].clone(), NeuronLocation::Hidden(i))
            }
            _ => {
                let i = rng.gen_range(0..self.output_layer.len());
                (self.output_layer[i].clone(), NeuronLocation::Output(i))
            }
        }
    }

    fn delete_neuron(&mut self, loc: NeuronLocation) -> NeuronTopology {
        if !loc.is_hidden() {
            panic!("Invalid neuron deletion");
        }

        let index = loc.unwrap();
        let neuron = Arc::into_inner(self.hidden_layers.remove(index)).unwrap();

        for n in &self.hidden_layers {
            let mut nw = n.write().unwrap();

            nw.inputs = nw
                .inputs
                .iter()
                .filter_map(|&(input_loc, w)| {
                    if !input_loc.is_hidden() {
                        return Some((input_loc, w));
                    }

                    if input_loc.unwrap() == index {
                        return None;
                    }

                    if input_loc.unwrap() > index {
                        return Some((NeuronLocation::Hidden(input_loc.unwrap() - 1), w));
                    }

                    Some((input_loc, w))
                })
                .collect();
        }

        for n2 in &self.output_layer {
            let mut nw = n2.write().unwrap();
            nw.inputs = nw
                .inputs
                .iter()
                .filter_map(|&(input_loc, w)| {
                    if !input_loc.is_hidden() {
                        return Some((input_loc, w));
                    }

                    if input_loc.unwrap() == index {
                        return None;
                    }

                    if input_loc.unwrap() > index {
                        return Some((NeuronLocation::Hidden(input_loc.unwrap() - 1), w));
                    }

                    Some((input_loc, w))
                })
                .collect();
        }

        neuron.into_inner().unwrap()
    }
}

// need to do all this manually because Arcs are cringe
impl<const I: usize, const O: usize> Clone for NeuralNetworkTopology<I, O> {
    fn clone(&self) -> Self {
        let input_layer = self
            .input_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(n.read().unwrap().clone())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = self
            .hidden_layers
            .iter()
            .map(|n| Arc::new(RwLock::new(n.read().unwrap().clone())))
            .collect();

        let output_layer = self
            .output_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(n.read().unwrap().clone())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_rate: self.mutation_rate,
            mutation_passes: self.mutation_passes,
        }
    }
}

impl<const I: usize, const O: usize> RandomlyMutable for NeuralNetworkTopology<I, O> {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        for _ in 0..self.mutation_passes {
            if rng.gen::<f32>() <= rate {
                // split preexisting connection
                let (mut n2, _) = self.rand_neuron(rng);

                while n2.read().unwrap().inputs.is_empty() {
                    (n2, _) = self.rand_neuron(rng);
                }

                let mut n2 = n2.write().unwrap();
                let i = rng.gen_range(0..n2.inputs.len());
                let (loc, w) = n2.inputs.remove(i);

                let loc3 = NeuronLocation::Hidden(self.hidden_layers.len());

                let n3 = NeuronTopology::new(vec![loc], ActivationScope::HIDDEN, rng);

                self.hidden_layers.push(Arc::new(RwLock::new(n3)));

                n2.inputs.insert(i, (loc3, w));
            }

            if rng.gen::<f32>() <= rate {
                // add a connection
                let (_, mut loc1) = self.rand_neuron(rng);
                let (_, mut loc2) = self.rand_neuron(rng);

                while loc1.is_output() || !self.add_connection(loc1, loc2, rng.gen::<f32>()) {
                    (_, loc1) = self.rand_neuron(rng);
                    (_, loc2) = self.rand_neuron(rng);
                }
            }

            if rng.gen::<f32>() <= rate && !self.hidden_layers.is_empty() {
                // remove a neuron
                let (_, mut loc) = self.rand_neuron(rng);

                while !loc.is_hidden() {
                    (_, loc) = self.rand_neuron(rng);
                }

                // delete the neuron
                self.delete_neuron(loc);
            }

            if rng.gen::<f32>() <= rate {
                // mutate a connection
                let (mut n, _) = self.rand_neuron(rng);

                while n.read().unwrap().inputs.is_empty() {
                    (n, _) = self.rand_neuron(rng);
                }

                let mut n = n.write().unwrap();
                let i = rng.gen_range(0..n.inputs.len());
                let (_, w) = &mut n.inputs[i];
                *w += rng.gen_range(-1.0..1.0) * rate;
            }

            if rng.gen::<f32>() <= rate {
                // mutate bias
                let (n, _) = self.rand_neuron(rng);
                let mut n = n.write().unwrap();

                n.bias += rng.gen_range(-1.0..1.0) * rate;
            }

            if rng.gen::<f32>() <= rate && !self.hidden_layers.is_empty() {
                // mutate activation function
                let reg = ACTIVATION_REGISTRY.read().unwrap();
                let activations = reg.activations_in_scope(ActivationScope::HIDDEN);

                let (mut n, mut loc) = self.rand_neuron(rng);

                while !loc.is_hidden() {
                    (n, loc) = self.rand_neuron(rng);
                }

                let mut nw = n.write().unwrap();

                // should probably not clone, but its not a huge efficiency issue anyways
                nw.activation = activations[rng.gen_range(0..activations.len())].clone();
            }
        }
    }
}

impl<const I: usize, const O: usize> DivisionReproduction for NeuralNetworkTopology<I, O> {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(self.mutation_rate, rng);
        child
    }
}

impl<const I: usize, const O: usize> PartialEq for NeuralNetworkTopology<I, O> {
    fn eq(&self, other: &Self) -> bool {
        if self.mutation_rate != other.mutation_rate
            || self.mutation_passes != other.mutation_passes
        {
            return false;
        }

        for i in 0..I {
            if *self.input_layer[i].read().unwrap() != *other.input_layer[i].read().unwrap() {
                return false;
            }
        }

        for i in 0..self.hidden_layers.len().min(other.hidden_layers.len()) {
            if *self.hidden_layers[i].read().unwrap() != *other.hidden_layers[i].read().unwrap() {
                return false;
            }
        }

        for i in 0..O {
            if *self.output_layer[i].read().unwrap() != *other.output_layer[i].read().unwrap() {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "serde")]
impl<const I: usize, const O: usize> From<nnt_serde::NNTSerde<I, O>>
    for NeuralNetworkTopology<I, O>
{
    fn from(value: nnt_serde::NNTSerde<I, O>) -> Self {
        let input_layer = value
            .input_layer
            .into_iter()
            .map(|n| Arc::new(RwLock::new(n)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value
            .hidden_layers
            .into_iter()
            .map(|n| Arc::new(RwLock::new(n)))
            .collect();

        let output_layer = value
            .output_layer
            .into_iter()
            .map(|n| Arc::new(RwLock::new(n)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        NeuralNetworkTopology {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_rate: value.mutation_rate,
            mutation_passes: value.mutation_passes,
        }
    }
}

#[cfg(feature = "crossover")]
impl<const I: usize, const O: usize> CrossoverReproduction for NeuralNetworkTopology<I, O> {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let input_layer = self
            .input_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(n.read().unwrap().clone())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut hidden_layers =
            Vec::with_capacity(self.hidden_layers.len().max(other.hidden_layers.len()));

        for i in 0..hidden_layers.len() {
            if rng.gen::<f32>() <= 0.5 {
                if let Some(n) = self.hidden_layers.get(i) {
                    let mut n = n.read().unwrap().clone();

                    n.inputs
                        .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
                    hidden_layers[i] = Arc::new(RwLock::new(n));

                    continue;
                }
            }

            let mut n = other.hidden_layers[i].read().unwrap().clone();

            n.inputs
                .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
            hidden_layers[i] = Arc::new(RwLock::new(n));
        }

        let mut output_layer: [Arc<RwLock<NeuronTopology>>; O] = self
            .output_layer
            .iter()
            .map(|n| Arc::new(RwLock::new(n.read().unwrap().clone())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        for (i, n) in self.output_layer.iter().enumerate() {
            if rng.gen::<f32>() <= 0.5 {
                let mut n = n.read().unwrap().clone();

                n.inputs
                    .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
                output_layer[i] = Arc::new(RwLock::new(n));

                continue;
            }

            let mut n = other.output_layer[i].read().unwrap().clone();

            n.inputs
                .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
            output_layer[i] = Arc::new(RwLock::new(n));
        }

        let mut child = Self {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_rate: self.mutation_rate,
            mutation_passes: self.mutation_passes,
        };

        child.mutate(self.mutation_rate, rng);

        child
    }
}

#[cfg(feature = "crossover")]
fn input_exists<const I: usize>(
    loc: NeuronLocation,
    input: &[Arc<RwLock<NeuronTopology>>; I],
    hidden: &[Arc<RwLock<NeuronTopology>>],
) -> bool {
    match loc {
        NeuronLocation::Input(i) => i < input.len(),
        NeuronLocation::Hidden(i) => i < hidden.len(),
        NeuronLocation::Output(_) => false,
    }
}

/// A stateless version of [`Neuron`][crate::Neuron].
#[derive(PartialEq, Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronTopology {
    /// The input locations and weights.
    pub inputs: Vec<(NeuronLocation, f32)>,

    /// The neuron's bias.
    pub bias: f32,

    /// The neuron's activation function.
    pub activation: ActivationFn,
}

impl NeuronTopology {
    /// Creates a new neuron with the given input locations.
    pub fn new(
        inputs: Vec<NeuronLocation>,
        current_scope: ActivationScope,
        rng: &mut impl Rng,
    ) -> Self {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        let activations = reg.activations_in_scope(current_scope);

        Self::new_with_activations(inputs, activations, rng)
    }

    /// Takes a collection of activation functions and chooses a random one to use.
    pub fn new_with_activations(
        inputs: Vec<NeuronLocation>,
        activations: impl IntoIterator<Item = ActivationFn>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut activations: Vec<_> = activations.into_iter().collect();

        Self::new_with_activation(
            inputs,
            activations.remove(rng.gen_range(0..activations.len())),
            rng,
        )
    }

    /// Creates a neuron with the activation.
    pub fn new_with_activation(
        inputs: Vec<NeuronLocation>,
        activation: ActivationFn,
        rng: &mut impl Rng,
    ) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|i| (i, rng.gen_range(-1.0..1.0)))
            .collect();

        Self {
            inputs,
            bias: rng.gen(),
            activation,
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
