use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

use atomic_float::AtomicF32;
use genetic_rs::prelude::*;
use map_macro::hash_set;
use rand::Rng;

use crate::{activation::*, activation_fn};

use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
use serde_big_array::BigArray;

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

// TODO serde
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralNetwork<const I: usize, const O: usize> {
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    pub input_layer: [Neuron; I],

    pub hidden_layers: Vec<Neuron>,

    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    pub output_layer: [Neuron; O],

    pub mutation_settings: MutationSettings,
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    // TODO option to set default output layer activations
    pub fn new(mutation_settings: MutationSettings, rng: &mut impl Rng) -> Self {
        let mut output_layer = Vec::with_capacity(O);

        for _ in 0..O {
            output_layer.push(Neuron::new_with_activation(
                hash_set! {},
                vec![],
                activation_fn!(sigmoid),
                rng,
            ));
        }

        let mut input_layer = Vec::with_capacity(I);

        for i in 0..I {
            let outputs = (0..rng.gen_range(1..=O))
                .map(|_| {
                    let mut already_chosen = Vec::new();
                    let mut j = rng.gen_range(0..O);
                    while already_chosen.contains(&j) {
                        j = rng.gen_range(0..O);
                    }

                    output_layer[j].inputs.insert(NeuronLocation::Input(i));
                    already_chosen.push(i);

                    (NeuronLocation::Output(i), rng.gen())
                })
                .collect();

            input_layer.push(Neuron::new(
                hash_set! {},
                outputs,
                ActivationScope::INPUT,
                rng,
            ));
        }

        let input_layer = input_layer.try_into().unwrap();
        let output_layer = output_layer.try_into().unwrap();

        Self {
            input_layer,
            hidden_layers: vec![],
            output_layer,
            mutation_settings,
        }
    }

    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        let cache = Arc::new(NeuralNetCache::from(self));
        cache.prime_inputs(inputs);

        (0..I)
            .into_par_iter()
            .for_each(|i| self.eval(NeuronLocation::Input(i), cache.clone()));

        cache.output()
    }

    fn eval(&self, loc: impl AsRef<NeuronLocation>, cache: Arc<NeuralNetCache<I, O>>) {
        let loc = loc.as_ref();

        if !cache.claim(loc) {
            // some other thread is already
            // waiting to do this task, currently doing it, or done.
            // no need to do it again.
            return;
        }

        while !cache.is_ready(loc) {
            // essentially spinlocks until the dependency tasks are complete,
            // while letting this thread do some work on random tasks.
            rayon::yield_now().unwrap();
        }

        let val = cache.get(loc);
        let n = self.get_neuron(loc);

        n.outputs.par_iter().for_each(|(loc2, weight)| {
            cache.add(loc2, n.activate(val * weight));
            self.eval(loc2, cache.clone());
        });
    }

    pub fn get_neuron(&self, loc: impl AsRef<NeuronLocation>) -> &Neuron {
        match loc.as_ref() {
            NeuronLocation::Input(i) => &self.input_layer[*i],
            NeuronLocation::Hidden(i) => &self.hidden_layers[*i],
            NeuronLocation::Output(i) => &self.output_layer[*i],
        }
    }

    pub fn get_neuron_mut(&mut self, loc: impl AsRef<NeuronLocation>) -> &mut Neuron {
        match loc.as_ref() {
            NeuronLocation::Input(i) => &mut self.input_layer[*i],
            NeuronLocation::Hidden(i) => &mut self.hidden_layers[*i],
            NeuronLocation::Output(i) => &mut self.output_layer[*i],
        }
    }

    pub fn split_connection(&mut self, connection: Connection, rng: &mut impl Rng) {
        let newloc = NeuronLocation::Hidden(self.hidden_layers.len());

        let a = self.get_neuron_mut(connection.from);
        let weight = a.remove_connection(connection.to).unwrap();

        a.outputs.push((newloc, weight));

        let n = Neuron::new(
            hash_set! { connection.from },
            vec![(connection.to, weight)],
            ActivationScope::HIDDEN,
            rng,
        );
        self.hidden_layers.push(n);
    }

    pub fn remove_connection(&mut self, connection: Connection) -> Option<f32> {
        let n = self.get_neuron_mut(connection.from);
        n.remove_connection(connection.to)
    }

    /// Adds a connection but does not check for cyclic linkages.
    pub unsafe fn add_connection_raw(&mut self, connection: Connection, weight: f32) {
        let a = self.get_neuron_mut(connection.from);
        a.outputs.push((connection.to, weight));

        let b = self.get_neuron_mut(connection.to);
        b.inputs.insert(connection.from);
    }

    /// Returns false if the connection is cyclic.
    pub fn is_connection_safe(&self, connection: Connection) -> bool {
        let mut visited = hash_set! { connection.from };

        self.dfs(&mut visited, connection.to)
    }

    // TODO maybe parallelize
    fn dfs(&self, visited: &mut HashSet<NeuronLocation>, current: NeuronLocation) -> bool {
        if !visited.insert(current) {
            return false;
        }

        let n = self.get_neuron(current);
        for (loc, _) in &n.outputs {
            if !self.dfs(visited, *loc) {
                return false;
            }
        }

        true
    }

    /// Safe, checked add connection method. Returns whether it performed the connection.
    pub fn add_connection(&mut self, connection: Connection, weight: f32) -> bool {
        if !self.is_connection_safe(connection) {
            return false;
        }

        unsafe { self.add_connection_raw(connection, weight); }

        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Connection {
    pub from: NeuronLocation,
    pub to: NeuronLocation,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neuron {
    pub inputs: HashSet<NeuronLocation>,
    pub outputs: Vec<(NeuronLocation, f32)>,
    pub bias: f32,
    pub activation_fn: ActivationFn,
}

impl Neuron {
    pub fn new_with_activation(
        inputs: HashSet<NeuronLocation>,
        outputs: Vec<(NeuronLocation, f32)>,
        activation_fn: ActivationFn,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            inputs,
            outputs,
            bias: rng.gen(),
            activation_fn,
        }
    }

    /// Creates a new neuron with the given output locations.
    pub fn new(
        inputs: HashSet<NeuronLocation>,
        outputs: Vec<(NeuronLocation, f32)>,
        current_scope: ActivationScope,
        rng: &mut impl Rng,
    ) -> Self {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        let activations = reg.activations_in_scope(current_scope);

        Self::new_with_activations(inputs, outputs, activations, rng)
    }

    /// Takes a collection of activation functions and chooses a random one to use.
    pub fn new_with_activations(
        inputs: HashSet<NeuronLocation>,
        outputs: Vec<(NeuronLocation, f32)>,
        activations: impl IntoIterator<Item = ActivationFn>,
        rng: &mut impl Rng,
    ) -> Self {
        // TODO get random in iterator form
        let mut activations: Vec<_> = activations.into_iter().collect();

        Self::new_with_activation(
            inputs,
            outputs,
            activations.remove(rng.gen_range(0..activations.len())),
            rng,
        )
    }

    pub fn activate(&self, v: f32) -> f32 {
        return self.activation_fn.func.activate(v);
    }

    pub fn get_weight(&self, output: impl AsRef<NeuronLocation>) -> Option<f32> {
        let loc = *output.as_ref();
        for out in &self.outputs {
            if out.0 == loc {
                return Some(out.1);
            }
        }

        None
    }

    pub fn remove_connection(&mut self, output: impl AsRef<NeuronLocation>) -> Option<f32> {
        let loc = *output.as_ref();
        let mut i = 0;

        while i < self.outputs.len() {
            if self.outputs[i].0 == loc {
                return Some(self.outputs.remove(i).1);
            }
            i += 1;
        }

        None
    }
}

/// A pseudo-pointer of sorts that is used for caching.
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

impl AsRef<NeuronLocation> for NeuronLocation {
    fn as_ref(&self) -> &NeuronLocation {
        self
    }
}

#[derive(Debug)]
pub struct NeuronCache {
    pub value: AtomicF32,
    pub expected_inputs: usize,
    pub total_inputs: AtomicUsize,
    pub claimed: AtomicBool,
}

impl From<&Neuron> for NeuronCache {
    fn from(value: &Neuron) -> Self {
        Self {
            value: AtomicF32::new(value.bias),
            expected_inputs: value.inputs.len(),
            total_inputs: AtomicUsize::new(0),
            claimed: AtomicBool::new(false),
        }
    }
}

#[derive(Debug)]
pub struct NeuralNetCache<const I: usize, const O: usize> {
    pub input_layer: [NeuronCache; I],
    pub hidden_layers: Vec<NeuronCache>,
    pub output_layer: [NeuronCache; O],
}

impl<const I: usize, const O: usize> NeuralNetCache<I, O> {
    pub fn get(&self, loc: impl AsRef<NeuronLocation>) -> f32 {
        match loc.as_ref() {
            NeuronLocation::Input(i) => self.input_layer[*i].value.load(Ordering::SeqCst),
            NeuronLocation::Hidden(i) => self.hidden_layers[*i].value.load(Ordering::SeqCst),
            NeuronLocation::Output(i) => self.output_layer[*i].value.load(Ordering::SeqCst),
        }
    }

    pub fn add(&self, loc: impl AsRef<NeuronLocation>, n: f32) -> f32 {
        match loc.as_ref() {
            NeuronLocation::Input(i) => self.input_layer[*i].value.fetch_add(n, Ordering::SeqCst),
            NeuronLocation::Hidden(i) => {
                let c = &self.hidden_layers[*i];
                let v = c.value.fetch_add(n, Ordering::SeqCst);
                c.total_inputs.fetch_add(1, Ordering::SeqCst);
                v
            }
            NeuronLocation::Output(i) => {
                let c = &self.output_layer[*i];
                let v = c.value.fetch_add(n, Ordering::SeqCst);
                c.total_inputs.fetch_add(1, Ordering::SeqCst);
                v
            }
        }
    }

    pub fn is_ready(&self, loc: impl AsRef<NeuronLocation>) -> bool {
        match loc.as_ref() {
            NeuronLocation::Input(i) => {
                let c = &self.input_layer[*i];
                c.expected_inputs >= c.total_inputs.load(Ordering::SeqCst)
            }
            NeuronLocation::Hidden(i) => {
                let c = &self.hidden_layers[*i];
                c.expected_inputs >= c.total_inputs.load(Ordering::SeqCst)
            }
            NeuronLocation::Output(i) => {
                let c = &self.output_layer[*i];
                c.expected_inputs >= c.total_inputs.load(Ordering::SeqCst)
            }
        }
    }

    pub fn prime_inputs(&self, inputs: [f32; I]) {
        for (i, v) in inputs.into_iter().enumerate() {
            self.input_layer[i].value.fetch_add(v, Ordering::SeqCst);
        }
    }

    pub fn output(&self) -> [f32; O] {
        let output: Vec<_> = self
            .output_layer
            .par_iter()
            .map(|c| c.value.load(Ordering::SeqCst))
            .collect();

        output.try_into().unwrap()
    }

    pub fn claim(&self, loc: impl AsRef<NeuronLocation>) -> bool {
        match loc.as_ref() {
            NeuronLocation::Input(i) => self.input_layer[*i]
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            NeuronLocation::Hidden(i) => self.hidden_layers[*i]
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            NeuronLocation::Output(i) => self.output_layer[*i]
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
        }
    }
}

impl<const I: usize, const O: usize> From<&NeuralNetwork<I, O>> for NeuralNetCache<I, O> {
    fn from(net: &NeuralNetwork<I, O>) -> Self {
        let input_layer: Vec<_> = net.input_layer.par_iter().map(|n| n.into()).collect();

        let input_layer = input_layer.try_into().unwrap();

        let hidden_layers: Vec<_> = net.hidden_layers.par_iter().map(|n| n.into()).collect();

        let hidden_layers = hidden_layers.try_into().unwrap();

        let output_layer: Vec<_> = net.output_layer.par_iter().map(|n| n.into()).collect();

        let output_layer = output_layer.try_into().unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}
