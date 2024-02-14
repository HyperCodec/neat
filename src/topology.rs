use std::{fmt, sync::{Arc, RwLock}};

use genetic_rs::prelude::*;
use rand::prelude::*;

/// Creates an [`ActivationFn`] object from a function
#[macro_export]
macro_rules! activation_fn {
    ($F: path) => {
        ActivationFn {
            func: Arc::new($F),
            name: String::from(stringify!($F)),
        }
    };

    {$($F: path),*} => {
        [$(activation_fn!($F)),*]
    };
}

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

    /// The mutation rate used in [`NeuralNetworkTopology::mutate`].
    pub mutation_rate: f32,

    /// The number of mutation passes (and thus, maximum number of possible mutations that can occur for each entity in the generation).
    pub mutation_passes: usize,
}

impl<const I: usize, const O: usize> NeuralNetworkTopology<I, O> {
    /// Creates a new [`NeuralNetworkTopology`].
    pub fn new(mutation_rate: f32, mutation_passes: usize, rng: &mut impl Rng) -> Self {
        let input_layer: [Arc<RwLock<NeuronTopology>>; I] = (0..I)
            .map(|_| Arc::new(RwLock::new(NeuronTopology::new_with_activation(vec![], activation_fn!(linear_activation), rng))))
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

            output_layer.push(Arc::new(RwLock::new(NeuronTopology::new_with_activation(input, activation_fn!(sigmoid), rng))));
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

    fn is_connection_cyclic(&self, loc1: NeuronLocation, loc2: NeuronLocation) -> bool {
        if loc1 == loc2 {
            return true;
        }

        for &(n, _w) in &self.get_neuron(loc1).read().unwrap().inputs {
            if self.is_connection_cyclic(n, loc2) {
                return true;
            }
        }

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
            1 => {
                if self.hidden_layers.is_empty() {
                    return self.rand_neuron(rng);
                }

                let i = rng.gen_range(0..self.hidden_layers.len());
                (self.hidden_layers[i].clone(), NeuronLocation::Hidden(i))
            }
            _ => {
                let i = rng.gen_range(0..self.output_layer.len());
                (self.output_layer[i].clone(), NeuronLocation::Output(i))
            }
        }
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

                let mut n3 = NeuronTopology::new(vec![loc], rng);
                n3.inputs[0].1 = 1.;

                self.hidden_layers
                    .push(Arc::new(RwLock::new(n3)));

                n2.inputs.insert(i, (loc3, w));
            }

            if rng.gen::<f32>() <= rate {
                // add a connection
                let (mut n1, mut loc1) = self.rand_neuron(rng);

                while n1.read().unwrap().inputs.is_empty() {
                    (n1, loc1) = self.rand_neuron(rng);
                }

                let (mut n2, mut loc2) = self.rand_neuron(rng);

                while self.is_connection_cyclic(loc1, loc2) {
                    (n2, loc2) = self.rand_neuron(rng);
                }

                n2.write().unwrap().inputs.push((loc1, rng.gen()));
            }

            if rng.gen::<f32>() <= rate && !self.hidden_layers.is_empty() {
                // remove a neuron
                let (_, mut loc) = self.rand_neuron(rng);

                while !loc.is_hidden() {
                    (_, loc) = self.rand_neuron(rng);
                }

                self.hidden_layers.remove(loc.unwrap());

                let mut done = false;
                'outer: for n in &self.hidden_layers {
                    let mut n2 = n.write().unwrap();

                    for (i, (loc2, _)) in n2.inputs.iter().enumerate() {
                        if loc == *loc2 {
                            n2.inputs.remove(i);
                            done = true;
                            break 'outer;
                        }
                    }
                }

                if !done {
                    'outer: for n in &self.output_layer {
                        let mut n2 = n.write().unwrap();
    
                        for (i, (loc2, _)) in n2.inputs.iter().enumerate() {
                            if loc == *loc2 {
                                n2.inputs.remove(i);
                                break 'outer;
                            }
                        }
                    }
                }
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
                let activations = activation_fn! {
                    sigmoid,
                    relu,
                    f32::tanh
                };

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

/*
#[cfg(feature = "crossover")]
impl CrossoverReproduction for NeuralNetworkTopology {
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        todo!();
    }
}
*/

/// An activation function object that implements [`fmt::Debug`] and is [`Send`]
#[derive(Clone)]
pub struct ActivationFn {
    /// The actual activation function.
    pub func: Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>,
    name: String,
}

impl fmt::Debug for ActivationFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)
    }
}

/// The sigmoid activation function.
pub fn sigmoid(n: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-n))
}

/// The ReLU activation function.
pub fn relu(n: f32) -> f32 {
    n.max(0.)
}

/// Activation function that does nothing.
pub fn linear_activation(n: f32) -> f32 {n}

/// A stateless version of [`Neuron`][crate::Neuron].
#[derive(Debug, Clone)]
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
    pub fn new(inputs: Vec<NeuronLocation>, rng: &mut impl Rng) -> Self {
        let activations = activation_fn! {
            sigmoid,
            relu,
            f32::tanh
        };

        Self::new_with_activations(inputs, activations, rng)
    }

    /// Takes a collection of activation functions and chooses a random one to use.
    pub fn new_with_activations(inputs: Vec<NeuronLocation>, activations: impl IntoIterator<Item = ActivationFn>, rng: &mut impl Rng) -> Self {
        let mut activations: Vec<_> = activations.into_iter().collect();

        Self::new_with_activation(inputs, activations.remove(rng.gen_range(0..activations.len())), rng)
    }

    /// Creates a neuron with the activation.
    pub fn new_with_activation(inputs: Vec<NeuronLocation>, activation: ActivationFn, rng: &mut impl Rng) -> Self {
        let inputs = inputs.into_iter().map(|i| (i, rng.gen_range(-1.0..1.0))).collect();

        Self {
            inputs,
            bias: rng.gen(),
            activation,
        }
    }
}

/// A pseudo-pointer of sorts used to make structural conversions very fast and easy to write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
