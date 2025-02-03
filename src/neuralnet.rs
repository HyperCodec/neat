use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

use atomic_float::AtomicF32;
use genetic_rs::prelude::*;
use rand::Rng;
use replace_with::replace_with_or_abort;

use crate::{
    activation::{self, builtin::*, *},
    activation_fn,
};

use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
use serde_big_array::BigArray;

/// The mutation settings for [`NeuralNetwork`].
/// Does not affect [`NeuralNetwork::mutate`], only [`NeuralNetwork::divide`] and [`NeuralNetwork::crossover`].
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MutationSettings {
    /// The chance of each mutation type to occur.
    pub mutation_rate: f32,

    /// The number of times to try to mutate the network.
    pub mutation_passes: usize,

    /// The maximum amount that the weights will be mutated by.
    pub weight_mutation_amount: f32,
}

impl Default for MutationSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.01,
            mutation_passes: 3,
            weight_mutation_amount: 0.5,
        }
    }
}

/// An abstract neural network type with `I` input neurons and `O` output neurons.
/// Hidden neurons are not organized into layers, but rather float and link freely
/// (or at least in any way that doesn't cause a cyclic dependency).
/// 
/// See [`NeuralNetwork::predict`] for usage.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralNetwork<const I: usize, const O: usize> {
    /// The input layer of neurons. Values specified in [`NeuralNetwork::predict`] will start here.
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    pub input_layer: [Neuron; I],

    /// The hidden layer(s) of neurons. They are not actually layered, but rather free-floating.
    pub hidden_layers: Vec<Neuron>,

    /// The output layer of neurons. Their values will be returned from [`NeuralNetwork::predict`].
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    pub output_layer: [Neuron; O],

    /// The mutation settings for the network.
    pub mutation_settings: MutationSettings,
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    // TODO option to set default output layer activations
    /// Creates a new random neural network with the given settings.
    pub fn new(mutation_settings: MutationSettings, rng: &mut impl Rng) -> Self {
        let mut output_layer = Vec::with_capacity(O);

        for _ in 0..O {
            output_layer.push(Neuron::new_with_activation(
                vec![],
                activation_fn!(sigmoid),
                rng,
            ));
        }

        let mut input_layer = Vec::with_capacity(I);

        for _ in 0..I {
            let mut already_chosen = Vec::new();
            let outputs = (0..rng.gen_range(1..=O))
                .map(|_| {
                    let mut j = rng.gen_range(0..O);
                    while already_chosen.contains(&j) {
                        j = rng.gen_range(0..O);
                    }

                    output_layer[j].input_count += 1;
                    already_chosen.push(j);

                    (NeuronLocation::Output(j), rng.gen())
                })
                .collect();

            input_layer.push(Neuron::new_with_activation(
                outputs,
                activation_fn!(linear_activation),
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

    /// Runs the neural network, propagating values from input to output layer.
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

        let loc = loc.as_ref();
        while !cache.is_ready(loc) {
            // essentially spinlocks until the dependency tasks are complete,
            // while letting this thread do some work on random tasks.
            rayon::yield_now();
        }

        let val = cache.get(loc);
        let n = self.get_neuron(loc);

        n.outputs.par_iter().for_each(|(loc2, weight)| {
            cache.add(loc2, n.activate(val * weight));
            self.eval(loc2, cache.clone());
        });
    }

    /// Get a neuron at the specified [`NeuronLocation`].
    pub fn get_neuron(&self, loc: impl AsRef<NeuronLocation>) -> &Neuron {
        match loc.as_ref() {
            NeuronLocation::Input(i) => &self.input_layer[*i],
            NeuronLocation::Hidden(i) => &self.hidden_layers[*i],
            NeuronLocation::Output(i) => &self.output_layer[*i],
        }
    }

    /// Get a mutable reference to the neuron at the specified [`NeuronLocation`].
    pub fn get_neuron_mut(&mut self, loc: impl AsRef<NeuronLocation>) -> &mut Neuron {
        match loc.as_ref() {
            NeuronLocation::Input(i) => &mut self.input_layer[*i],
            NeuronLocation::Hidden(i) => &mut self.hidden_layers[*i],
            NeuronLocation::Output(i) => &mut self.output_layer[*i],
        }
    }

    /// Split a [`Connection`] into two of the same weight, joined by a new [`Neuron`] in the hidden layer(s).
    pub fn split_connection(&mut self, connection: Connection, rng: &mut impl Rng) {
        let newloc = NeuronLocation::Hidden(self.hidden_layers.len());

        let a = self.get_neuron_mut(connection.from);
        let weight = unsafe { a.remove_connection(connection.to) }.unwrap();

        a.outputs.push((newloc, weight));

        let n = Neuron::new(vec![(connection.to, weight)], NeuronScope::HIDDEN, rng);
        self.hidden_layers.push(n);
    }

    /// Adds a connection but does not check for cyclic linkages.
    /// Marked as unsafe because it could cause a hang/livelock when predicting due to cyclic linkage.
    /// There is no actual UB or unsafe code associated with it.
    pub unsafe fn add_connection_raw(&mut self, connection: Connection, weight: f32) {
        let a = self.get_neuron_mut(connection.from);
        a.outputs.push((connection.to, weight));

        // let b = self.get_neuron_mut(connection.to);
        // b.inputs.insert(connection.from);
    }

    /// Returns false if the connection is cyclic.
    pub fn is_connection_safe(&self, connection: Connection) -> bool {
        let mut visited = HashSet::from([connection.from]);

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

    /// Safe, checked add connection method. Returns false if it aborted connecting due to cyclic linkage.
    pub fn add_connection(&mut self, connection: Connection, weight: f32) -> bool {
        if !self.is_connection_safe(connection) {
            return false;
        }

        unsafe {
            self.add_connection_raw(connection, weight);
        }

        true
    }

    /// Mutates a connection's weight.
    pub fn mutate_weight(&mut self, connection: Connection, rng: &mut impl Rng) {
        let rate = self.mutation_settings.weight_mutation_amount;
        let n = self.get_neuron_mut(connection.from);
        n.mutate_weight(connection.to, rate, rng).unwrap();
    }

    /// Get a random valid location within the network.
    pub fn random_location(&self, rng: &mut impl Rng) -> NeuronLocation {
        match rng.gen_range(0..3) {
            0 => NeuronLocation::Input(rng.gen_range(0..self.input_layer.len())),
            1 => NeuronLocation::Hidden(rng.gen_range(0..self.hidden_layers.len())),
            2 => NeuronLocation::Output(rng.gen_range(0..self.output_layer.len())),
            _ => unreachable!(),
        }
    }

    /// Get a random valid location within a [`NeuronScope`].
    pub fn random_location_in_scope(
        &self,
        rng: &mut impl Rng,
        scope: NeuronScope,
    ) -> NeuronLocation {
        let loc = self.random_location(rng);

        // this is a lazy and slow way of donig it, TODO better version.
        if !scope.contains(NeuronScope::from(loc)) {
            return self.random_location_in_scope(rng, scope);
        }

        loc
    }

    /// Remove a connection and any hanging neurons caused by the deletion.
    /// Returns whether there was a hanging neuron.
    pub fn remove_connection(&mut self, connection: Connection) -> bool {
        let a = self.get_neuron_mut(connection.from);
        unsafe { a.remove_connection(connection.to) }.unwrap();

        let b = self.get_neuron_mut(connection.to);
        b.input_count -= 1;

        if b.input_count <= 0 {
            self.remove_neuron(connection.to);
            return true;
        }

        false
    }

    /// Remove a neuron and downshift all connection indexes to compensate for it.
    pub fn remove_neuron(&mut self, loc: impl AsRef<NeuronLocation>) {
        let loc = loc.as_ref();
        if !loc.is_hidden() {
            panic!("Can only remove neurons from hidden layer");
        }

        unsafe {
            self.downshift_connections(loc.unwrap());
        }
    }

    unsafe fn downshift_connections(&mut self, i: usize) {
        self.input_layer
            .par_iter_mut()
            .for_each(|n| n.downshift_outputs(i));

        self.hidden_layers
            .par_iter_mut()
            .for_each(|n| n.downshift_outputs(i));
    }

    // TODO maybe more parallelism and pass Connection info.
    /// Runs the `callback` on the weights of the neural network in parallel, allowing it to modify weight values.
    pub fn map_weights(&mut self, callback: impl Fn(&mut f32) + Sync) {
        for n in &mut self.input_layer {
            n.outputs.par_iter_mut().for_each(|(_, w)| callback(w));
        }

        for n in &mut self.hidden_layers {
            n.outputs.par_iter_mut().for_each(|(_, w)| callback(w));
        }
    }

    unsafe fn clear_input_counts(&mut self) {
        // not sure whether all this parallelism is necessary or if it will just generate overhead
        // rayon::scope(|s| {
        //     s.spawn(|_| self.input_layer.par_iter_mut().for_each(|n| n.input_count = 0));
        //     s.spawn(|_| self.hidden_layers.par_iter_mut().for_each(|n| n.input_count = 0));
        //     s.spawn(|_| self.output_layer.par_iter_mut().for_each(|n| n.input_count = 0));
        // });

        self.input_layer
            .par_iter_mut()
            .for_each(|n| n.input_count = 0);
        self.hidden_layers
            .par_iter_mut()
            .for_each(|n| n.input_count = 0);
        self.output_layer
            .par_iter_mut()
            .for_each(|n| n.input_count = 0);
    }

    /// Recalculates the [`input_count`][`Neuron::input_count`] field for all neurons in the network.
    pub fn recalculate_input_counts(&mut self) {
        unsafe { self.clear_input_counts() };

        for i in 0..I {
            for j in 0..self.input_layer[i].outputs.len() {
                let (loc, _) = self.input_layer[i].outputs[j];
                self.get_neuron_mut(loc).input_count += 1;
            }
        }

        for i in 0..self.hidden_layers.len() {
            for j in 0..self.hidden_layers[i].outputs.len() {
                let (loc, _) = self.hidden_layers[i].outputs[j];
                self.get_neuron_mut(loc).input_count += 1;
            }
        }
    }
}

impl<const I: usize, const O: usize> RandomlyMutable for NeuralNetwork<I, O> {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
            if rng.gen::<f32>() <= rate {
                // split connection
                let from = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
                let n = self.get_neuron(from);
                let (to, _) = n.random_output(rng);

                self.split_connection(Connection { from, to }, rng);
            }

            if rng.gen::<f32>() <= rate {
                // add connection
                let weight = rng.gen::<f32>();

                let from = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
                let to = self.random_location_in_scope(rng, !NeuronScope::INPUT);

                let mut connection = Connection { from, to };
                while !self.add_connection(connection, weight) {
                    let from = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
                    let to = self.random_location_in_scope(rng, !NeuronScope::INPUT);
                    connection = Connection { from, to };
                }
            }

            if rng.gen::<f32>() <= rate {
                // remove connection

                let from = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
                let a = self.get_neuron(from);
                let (to, _) = a.random_output(rng);

                self.remove_connection(Connection { from, to });
            }

            self.map_weights(|w| {
                // TODO maybe `Send`able rng.
                let mut rng = rand::thread_rng();

                if rng.gen::<f32>() <= rate {
                    *w += rng.gen_range(-rate..rate);
                }
            });
        }
}

impl<const I: usize, const O: usize> DivisionReproduction for NeuralNetwork<I, O> {
    fn divide(&self, rng: &mut impl Rng) -> Self {
        let mut child = self.clone();

        for _ in 0..self.mutation_settings.mutation_passes {
            child.mutate(child.mutation_settings.mutation_rate, rng);
        }

        child
    }
}

impl<const I: usize, const O: usize> CrossoverReproduction for NeuralNetwork<I, O> {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let mut output_layer = self.output_layer.clone();

        for (i, n) in output_layer.iter_mut().enumerate() {
            if rng.gen::<f32>() >= 0.5 {
                *n = other.output_layer[i].clone();
            }
        }

        let hidden_len = self.hidden_layers.len().max(other.hidden_layers.len());
        let mut hidden_layers = Vec::with_capacity(hidden_len);

        for i in 0..hidden_len {
            if rng.gen::<f32>() >= 0.5 {
                if let Some(n) = self.hidden_layers.get(i) {
                    let mut n = n.clone();
                    n.prune_invalid_outputs(hidden_len, O);

                    hidden_layers[i] = n;

                    continue;
                }
            }

            let mut n = other.hidden_layers[i].clone();
            n.prune_invalid_outputs(hidden_len, O);

            hidden_layers[i] = n;
        }

        let mut input_layer = self.input_layer.clone();

        for (i, n) in input_layer.iter_mut().enumerate() {
            if rng.gen::<f32>() >= 0.5 {
                *n = other.input_layer[i].clone();
            }
            n.prune_invalid_outputs(hidden_len, O);
        }

        // crossover mutation settings just in case.
        let mutation_settings = if rng.gen::<f32>() >= 0.5 {
            self.mutation_settings.clone()
        } else {
            other.mutation_settings.clone()
        };

        let mut child = Self {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_settings,
        };

        // TODO maybe find a way to do this while doing crossover stuff instead of recalculating everything.
        // would be annoying to implement though.
        child.recalculate_input_counts();

        for _ in 0..child.mutation_settings.mutation_passes {
            child.mutate(child.mutation_settings.mutation_rate, rng);
        }

        child
    }
}

fn output_exists(loc: NeuronLocation, hidden_len: usize, output_len: usize) -> bool {
    match loc {
        NeuronLocation::Input(_) => false,
        NeuronLocation::Hidden(i) => i < hidden_len,
        NeuronLocation::Output(i) => i < output_len,
    }
}

/// A helper struct for operations on connections between neurons.
/// It does not contain information about the weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Connection {
    /// The source of the connection.
    pub from: NeuronLocation,

    /// The destination of the connection.
    pub to: NeuronLocation,
}

/// A stateless neuron. Contains info about bias, activation, and connections.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neuron {
    /// The input count used in [`NeuralNetCache`]. Not safe to modify.
    pub input_count: usize,

    /// The connections and weights to other neurons.
    pub outputs: Vec<(NeuronLocation, f32)>,

    /// The initial value of the neuron.
    pub bias: f32,

    /// The activation function applied to the value before propagating to [`outputs`][Neuron::outputs].
    pub activation_fn: ActivationFn,
}

impl Neuron {
    /// Creates a new neuron with a specified activation function and outputs.
    pub fn new_with_activation(
        outputs: Vec<(NeuronLocation, f32)>,
        activation_fn: ActivationFn,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            input_count: 0,
            outputs,
            bias: rng.gen(),
            activation_fn,
        }
    }

    /// Creates a new neuron with the given output locations.
    /// Chooses a random activation function within the specified scope.
    pub fn new(
        outputs: Vec<(NeuronLocation, f32)>,
        current_scope: NeuronScope,
        rng: &mut impl Rng,
    ) -> Self {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        let activations = reg.activations_in_scope(current_scope);

        Self::new_with_activations(outputs, activations, rng)
    }

    /// Creates a new neuron with the given outputs.
    /// Takes a collection of activation functions and chooses a random one from them to use.
    pub fn new_with_activations(
        outputs: Vec<(NeuronLocation, f32)>,
        activations: impl IntoIterator<Item = ActivationFn>,
        rng: &mut impl Rng,
    ) -> Self {
        // TODO get random in iterator form
        let mut activations: Vec<_> = activations.into_iter().collect();

        // TODO maybe Result instead.
        if activations.is_empty() {
            panic!("Empty activations list provided");
        }

        Self::new_with_activation(
            outputs,
            activations.remove(rng.gen_range(0..activations.len())),
            rng,
        )
    }

    /// Runs the [activation function][Neuron::activation_fn] on the given value and returns it.
    pub fn activate(&self, v: f32) -> f32 {
        self.activation_fn.func.activate(v)
    }

    /// Get the weight of the provided output location. Returns `None` if not found.
    pub fn get_weight(&self, output: impl AsRef<NeuronLocation>) -> Option<f32> {
        let loc = *output.as_ref();
        for out in &self.outputs {
            if out.0 == loc {
                return Some(out.1);
            }
        }

        None
    }

    /// Tries to remove a connection from the neuron and returns the weight if it was found. 
    /// Marked as unsafe because it will not update the destination's [`input_count`][Neuron::input_count].
    pub unsafe fn remove_connection(&mut self, output: impl AsRef<NeuronLocation>) -> Option<f32> {
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

    /// Randomly mutates the specified weight with the rate.
    pub fn mutate_weight(
        &mut self,
        output: impl AsRef<NeuronLocation>,
        rate: f32,
        rng: &mut impl Rng,
    ) -> Option<f32> {
        let loc = *output.as_ref();
        let mut i = 0;

        while i < self.outputs.len() {
            let o = &mut self.outputs[i];
            if o.0 == loc {
                o.1 += rng.gen_range(-rate..rate);

                return Some(o.1);
            }

            i += 1;
        }

        None
    }

    /// Get a random output location and weight.
    pub fn random_output(&self, rng: &mut impl Rng) -> (NeuronLocation, f32) {
        self.outputs[rng.gen_range(0..self.outputs.len())]
    }

    pub(crate) fn downshift_outputs(&mut self, i: usize) {
        // TODO par_iter_mut instead of replace
        replace_with_or_abort(&mut self.outputs, |o| {
            o.into_par_iter()
                .map(|(loc, w)| match loc {
                    NeuronLocation::Hidden(j) if j > i => (NeuronLocation::Hidden(j - 1), w),
                    _ => (loc, w),
                })
                .collect()
        });
    }

    /// Removes any outputs pointing to a nonexistent neuron.
    pub fn prune_invalid_outputs(&mut self, hidden_len: usize, output_len: usize) {
        self.outputs
            .retain(|(loc, _)| output_exists(*loc, hidden_len, output_len));
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

/// Handles the state of a single neuron for [`NeuralNetCache`].
#[derive(Debug, Default)]
pub struct NeuronCache {
    /// The value of the neuron.
    pub value: AtomicF32,

    /// The expected input count.
    pub expected_inputs: usize,

    /// The number of inputs that have finished evaluating.
    pub finished_inputs: AtomicUsize,

    /// Whether or not a thread has claimed this neuron to work on it.
    pub claimed: AtomicBool,
}

impl NeuronCache {
    /// Creates a new [`NeuronCache`] given relevant info.
    /// Use [`NeuronCache::from`] instead to create cache for a [`Neuron`].
    pub fn new(bias: f32, expected_inputs: usize) -> Self {
        Self {
            value: AtomicF32::new(bias),
            expected_inputs,
            ..Default::default()
        }
    }
}

impl From<&Neuron> for NeuronCache {
    fn from(value: &Neuron) -> Self {
        Self {
            value: AtomicF32::new(value.bias),
            expected_inputs: value.input_count,
            finished_inputs: AtomicUsize::new(0),
            claimed: AtomicBool::new(false),
        }
    }
}

/// A cache type used in [`NeuralNetwork::predict`] to track state.
#[derive(Debug)]
pub struct NeuralNetCache<const I: usize, const O: usize> {
    /// The input layer cache.
    pub input_layer: [NeuronCache; I],

    /// The hidden layer(s) cache.
    pub hidden_layers: Vec<NeuronCache>,

    /// The output layer cache.
    pub output_layer: [NeuronCache; O],
}

impl<const I: usize, const O: usize> NeuralNetCache<I, O> {
    /// Gets the value of a neuron at the given location.
    pub fn get(&self, loc: impl AsRef<NeuronLocation>) -> f32 {
        match loc.as_ref() {
            NeuronLocation::Input(i) => self.input_layer[*i].value.load(Ordering::SeqCst),
            NeuronLocation::Hidden(i) => self.hidden_layers[*i].value.load(Ordering::SeqCst),
            NeuronLocation::Output(i) => self.output_layer[*i].value.load(Ordering::SeqCst),
        }
    }

    /// Adds a value to the neuron at the specified location and increments [`finished_inputs`][NeuronCache::finished_inputs].
    pub fn add(&self, loc: impl AsRef<NeuronLocation>, n: f32) -> f32 {
        match loc.as_ref() {
            NeuronLocation::Input(i) => self.input_layer[*i].value.fetch_add(n, Ordering::SeqCst),
            NeuronLocation::Hidden(i) => {
                let c = &self.hidden_layers[*i];
                let v = c.value.fetch_add(n, Ordering::SeqCst);
                c.finished_inputs.fetch_add(1, Ordering::SeqCst);
                v
            }
            NeuronLocation::Output(i) => {
                let c = &self.output_layer[*i];
                let v = c.value.fetch_add(n, Ordering::SeqCst);
                c.finished_inputs.fetch_add(1, Ordering::SeqCst);
                v
            }
        }
    }

    /// Returns whether [`finished_inputs`][NeuronCache::finished_inputs] matches [`expected_inputs`][NeuronCache::expected_inputs].
    pub fn is_ready(&self, loc: impl AsRef<NeuronLocation>) -> bool {
        match loc.as_ref() {
            NeuronLocation::Input(i) => {
                let c = &self.input_layer[*i];
                c.expected_inputs >= c.finished_inputs.load(Ordering::SeqCst)
            }
            NeuronLocation::Hidden(i) => {
                let c = &self.hidden_layers[*i];
                c.expected_inputs >= c.finished_inputs.load(Ordering::SeqCst)
            }
            NeuronLocation::Output(i) => {
                let c = &self.output_layer[*i];
                c.expected_inputs >= c.finished_inputs.load(Ordering::SeqCst)
            }
        }
    }

    /// Adds the input values to the input layer of neurons.
    pub fn prime_inputs(&self, inputs: [f32; I]) {
        for (i, v) in inputs.into_iter().enumerate() {
            self.input_layer[i].value.fetch_add(v, Ordering::SeqCst);
        }
    }

    /// Fetches and packs the output layer values into an array.
    pub fn output(&self) -> [f32; O] {
        let output: Vec<_> = self
            .output_layer
            .par_iter()
            .map(|c| c.value.load(Ordering::SeqCst))
            .collect();

        output.try_into().unwrap()
    }

    /// Attempts to claim a neuron. Returns false if it has already been claimed.
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
