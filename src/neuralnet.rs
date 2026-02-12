use std::{
    collections::{HashMap, HashSet, VecDeque},
    ops::{Index, IndexMut},
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
    activation::{builtin::*, *},
    activation_fn,
};

use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde")]
use serde_big_array::BigArray;

#[cfg(feature = "serde")]
mod outputs_serde {
    use super::*;
    use std::collections::HashMap;

    pub fn serialize<S>(
        map: &HashMap<NeuronLocation, f32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vec: Vec<(NeuronLocation, f32)> = map.iter().map(|(k, v)| (*k, *v)).collect();
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<NeuronLocation, f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<(NeuronLocation, f32)> = Vec::deserialize(deserializer)?;
        Ok(vec.into_iter().collect())
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
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    // TODO option to set default output layer activations
    /// Creates a new random neural network with the given settings.
    pub fn new(rng: &mut impl rand::Rng) -> Self {
        let mut output_layer = Vec::with_capacity(O);

        for _ in 0..O {
            output_layer.push(Neuron::new_with_activation(
                HashMap::new(),
                activation_fn!(sigmoid),
                rng,
            ));
        }

        let mut input_layer = Vec::with_capacity(I);

        for _ in 0..I {
            let mut already_chosen = HashSet::new();
            let num_outputs = rng.random_range(1..=O);
            let mut outputs = HashMap::new();

            for _ in 0..num_outputs {
                let mut j = rng.random_range(0..O);
                while already_chosen.contains(&j) {
                    j = rng.random_range(0..O);
                }

                output_layer[j].input_count += 1;
                already_chosen.insert(j);

                outputs.insert(NeuronLocation::Output(j), rng.random());
            }

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
        }
    }

    /// Runs the neural network, propagating values from input to output layer.
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        let cache = Arc::new(NeuralNetCache::from(self));
        cache.prime_inputs(inputs);

        (0..I)
            .into_par_iter()
            .for_each(|i| self.eval(NeuronLocation::Input(i), cache.clone()));

        let mut outputs = [0.0; O];
        for (i, output) in outputs.iter_mut().enumerate().take(O) {
            let n = &self.output_layer[i];
            let val = cache.get(NeuronLocation::Output(i));
            *output = n.activate(val);
        }

        outputs
    }

    fn eval(&self, loc: NeuronLocation, cache: Arc<NeuralNetCache<I, O>>) {
        if !cache.claim(loc) {
            // some other thread is already
            // waiting to do this task, currently doing it, or done.
            // no need to do it again.
            return;
        }

        while !cache.is_ready(loc) {
            // essentially spinlocks until the dependency tasks are complete,
            // while letting this thread do some work on random tasks.
            rayon::yield_now();
        }

        let n = &self[loc];
        let val = n.activate(cache.get(loc));

        n.outputs.par_iter().for_each(|(&loc2, weight)| {
            cache.add(loc2, val * weight);
            self.eval(loc2, cache.clone());
        });
    }

    /// Get a neuron at the specified [`NeuronLocation`].
    pub fn get_neuron(&self, loc: NeuronLocation) -> Option<&Neuron> {
        if !self.neuron_exists(loc) {
            None
        } else {
            Some(&self[loc])
        }
    }

    /// Returns whether there is a neuron at the location
    pub fn neuron_exists(&self, loc: NeuronLocation) -> bool {
        match loc {
            NeuronLocation::Input(i) => i < I,
            NeuronLocation::Hidden(i) => i < self.hidden_layers.len(),
            NeuronLocation::Output(i) => i < O,
        }
    }

    /// Get a mutable reference to the neuron at the specified [`NeuronLocation`].
    pub fn get_neuron_mut(&mut self, loc: NeuronLocation) -> Option<&mut Neuron> {
        if !self.neuron_exists(loc) {
            None
        } else {
            Some(&mut self[loc])
        }
    }

    /// Adds a new neuron to hidden layer. Updates [`input_count`][Neuron::input_count]s automatically.
    /// Removes any output connections that point to invalid neurons or would result in cyclic linkage.
    /// Returns whether all output connections were valid.
    /// Due to the cyclic check, this function has time complexity O(nm), where n is the number of neurons
    /// and m is the number of output connections.
    pub fn add_neuron(&mut self, mut n: Neuron) -> bool {
        let mut valid = true;
        let new_loc = NeuronLocation::Hidden(self.hidden_layers.len());
        let outputs = n.outputs.keys().cloned().collect::<Vec<_>>();
        for loc in outputs {
            if !self.neuron_exists(loc)
                || !self.is_connection_safe(Connection {
                    from: new_loc,
                    to: loc,
                })
            {
                n.outputs.remove(&loc);
                valid = false;
                continue;
            }

            let n = &mut self[loc];
            n.input_count += 1;
        }

        self.hidden_layers.push(n);

        valid
    }

    /// Split a [`Connection`] into two of the same weight, joined by a new [`Neuron`] in the hidden layer(s).
    pub fn split_connection(&mut self, connection: Connection, rng: &mut impl Rng) {
        let new_loc = NeuronLocation::Hidden(self.hidden_layers.len());

        let a = &mut self[connection.from];
        let w = a
            .outputs
            .remove(&connection.to)
            .expect("invalid connection.to");

        a.outputs.insert(new_loc, w);

        let mut outputs = HashMap::new();
        outputs.insert(connection.to, w);
        let mut new_n = Neuron::new(outputs, NeuronScope::HIDDEN, rng);
        new_n.input_count = 1;
        self.hidden_layers.push(new_n);
    }

    /// Adds a connection but does not check for cyclic linkages.
    pub fn add_connection_unchecked(&mut self, connection: Connection, weight: f32) {
        let a = &mut self[connection.from];
        a.outputs.insert(connection.to, weight);

        let b = &mut self[connection.to];
        b.input_count += 1;
    }

    /// Returns false if the connection is cyclic or the input/output neurons are otherwise invalid in some other way.
    /// Can be O(n) over the number of neurons in the network.
    pub fn is_connection_safe(&self, connection: Connection) -> bool {
        if connection.from.is_output()
            || connection.to.is_input()
            || connection.from == connection.to
            || (self.neuron_exists(connection.from)
                && self[connection.from].outputs.contains_key(&connection.to))
        {
            return false;
        }
        let mut visited = HashSet::from([connection.from]);
        self.dfs(&mut visited, connection.to)
    }

    fn dfs(&self, visited: &mut HashSet<NeuronLocation>, current: NeuronLocation) -> bool {
        if !visited.insert(current) {
            return false;
        }

        let n = &self[current];
        for loc in n.outputs.keys() {
            if !self.dfs(visited, *loc) {
                return false;
            }
        }

        true
    }

    /// Safe, checked add connection method. Returns false if it aborted due to cyclic linkage.
    /// Note that checking for cyclic linkage is O(n) over all neurons in the network, which
    /// may be expensive for larger networks.
    pub fn add_connection(&mut self, connection: Connection, weight: f32) -> bool {
        if !self.is_connection_safe(connection) {
            return false;
        }

        self.add_connection_unchecked(connection, weight);

        true
    }

    /// Attempts to add a random connection, retrying if unsafe.
    /// Returns the connection if it established one before reaching max_retries.
    pub fn add_random_connection(
        &mut self,
        max_retries: usize,
        rng: &mut impl rand::Rng,
    ) -> Option<Connection> {
        for _ in 0..max_retries {
            let a = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
            let b = self.random_location_in_scope(rng, !NeuronScope::INPUT);

            let conn = Connection { from: a, to: b };
            if self.add_connection(conn, rng.random()) {
                return Some(conn);
            }
        }

        None
    }

    /// Attempts to get a random connection, retrying if the neuron it found
    /// doesn't have any outbound connections.
    /// Returns the connection if it found one before reaching max_retries.
    pub fn get_random_connection(
        &mut self,
        max_retries: usize,
        rng: &mut impl rand::Rng,
    ) -> Option<Connection> {
        for _ in 0..max_retries {
            let a = self.random_location_in_scope(rng, !NeuronScope::OUTPUT);
            let an = &self[a];
            if an.outputs.is_empty() {
                continue;
            }

            let mut iter = an
                .outputs
                .keys()
                .skip(rng.random_range(0..an.outputs.len()));
            let b = iter.next().unwrap();

            let conn = Connection { from: a, to: *b };
            return Some(conn);
        }

        None
    }

    /// Attempts to remove a random connection, retrying if the neuron it found
    /// doesn't have any outbound connections. Also removes hanging neurons created
    /// by removing the connection.
    ///
    /// Returns the connection if it removed one before reaching max_retries.
    pub fn remove_random_connection(
        &mut self,
        max_retries: usize,
        rng: &mut impl rand::Rng,
    ) -> Option<Connection> {
        if let Some(conn) = self.get_random_connection(max_retries, rng) {
            self.remove_connection(conn);
            Some(conn)
        } else {
            None
        }
    }

    /// Mutates a connection's weight.
    pub fn mutate_weight(&mut self, connection: Connection, amount: f32, rng: &mut impl Rng) {
        let n = &mut self[connection.from];
        n.mutate_weight(connection.to, amount, rng).unwrap();
    }

    /// Get a random valid location within the network.
    pub fn random_location(&self, rng: &mut impl Rng) -> NeuronLocation {
        if self.hidden_layers.is_empty() {
            if rng.random_range(0..=1) != 0 {
                return NeuronLocation::Input(rng.random_range(0..I));
            }
            return NeuronLocation::Output(rng.random_range(0..O));
        }

        match rng.random_range(0..3) {
            0 => NeuronLocation::Input(rng.random_range(0..I)),
            1 => NeuronLocation::Hidden(rng.random_range(0..self.hidden_layers.len())),
            2 => NeuronLocation::Output(rng.random_range(0..O)),
            _ => unreachable!(),
        }
    }

    /// Get a random valid location within a [`NeuronScope`].
    pub fn random_location_in_scope(
        &self,
        rng: &mut impl rand::Rng,
        scope: NeuronScope,
    ) -> NeuronLocation {
        if scope == NeuronScope::NONE {
            panic!("cannot select from empty scope");
        }

        let mut layers = Vec::with_capacity(3);
        if scope.contains(NeuronScope::INPUT) {
            layers.push((NeuronLocation::Input(0), I));
        }
        if scope.contains(NeuronScope::HIDDEN) && !self.hidden_layers.is_empty() {
            layers.push((NeuronLocation::Hidden(0), self.hidden_layers.len()));
        }
        if scope.contains(NeuronScope::OUTPUT) {
            layers.push((NeuronLocation::Output(0), O));
        }

        let (mut loc, size) = layers[rng.random_range(0..layers.len())];
        loc.set_inner(rng.random_range(0..size));
        loc
    }

    /// Remove a connection and indicate whether the destination neuron became hanging
    /// (with the exception of output layer neurons).
    /// Returns `true` if the destination neuron has input_count == 0 and should be removed.
    /// Callers must handle the removal of the destination neuron if needed.
    pub fn remove_connection_raw(&mut self, connection: Connection) -> bool {
        let a = self
            .get_neuron_mut(connection.from)
            .expect("invalid connection.from");
        if a.outputs.remove(&connection.to).is_none() {
            panic!("invalid connection.to");
        }

        let b = &mut self[connection.to];

        // if the invariants held at the beginning of the call,
        // this should never underflow, but some cases like remove_cycles
        // may temporarily break invariants.
        b.input_count = b.input_count.saturating_sub(1);

        // signal removal
        connection.to.is_hidden() && b.input_count == 0
    }

    /// Remove a connection from the network.
    /// This will also deal with hanging neurons iteratively to avoid recursion that
    /// can invalidate stored indices during nested deletions.
    /// This method is preferable to [`remove_connection_raw`][NeuralNetwork::remove_connection_raw] for a majority of usecases,
    /// as it preserves the invariants of the neural network.
    pub fn remove_connection(&mut self, conn: Connection) -> bool {
        if self.remove_connection_raw(conn) {
            self.remove_neuron(conn.to);
            return true;
        }
        false
    }

    /// Remove a neuron and downshift all connection indices to compensate for it.
    /// Returns the number of neurons removed that were under the index of the removed neuron (including itself).
    /// This will also deal with hanging neurons iteratively to avoid recursion that
    /// can invalidate stored indices during nested deletions.
    pub fn remove_neuron(&mut self, loc: NeuronLocation) -> usize {
        if !loc.is_hidden() {
            panic!("cannot remove neurons in input or output layer");
        }

        let initial_i = loc.unwrap();

        let mut work = VecDeque::new();
        work.push_back(loc);

        let mut removed = 0;
        while let Some(cur_loc) = work.pop_front() {
            // if the neuron was already removed due to earlier deletions, skip.
            // i don't think it realistically should ever happen, but just in case.
            if !self.neuron_exists(cur_loc) {
                continue;
            }

            let outputs = {
                let n = &self[cur_loc];
                n.outputs.keys().cloned().collect::<Vec<_>>()
            };

            for target in outputs {
                if self.remove_connection_raw(Connection {
                    from: cur_loc,
                    to: target,
                }) {
                    // target became hanging; schedule it for removal.
                    work.push_back(target);
                }
            }

            // Re-check that the neuron still exists and is hidden before removing.
            if !self.neuron_exists(cur_loc) || !cur_loc.is_hidden() {
                continue;
            }

            let i = cur_loc.unwrap();
            if i < self.hidden_layers.len() {
                self.hidden_layers.remove(i);
                if i <= initial_i {
                    removed += 1;
                }
                self.downshift_connections(i, &mut work); // O(n^2) bad, but we can optimize later if it's a problem.
            }
        }

        removed
    }

    fn downshift_connections(&mut self, i: usize, work: &mut VecDeque<NeuronLocation>) {
        self.input_layer
            .par_iter_mut()
            .for_each(|n| n.downshift_outputs(i));

        self.hidden_layers
            .par_iter_mut()
            .for_each(|n| n.downshift_outputs(i));

        work.par_iter_mut().for_each(|loc| match loc {
            NeuronLocation::Hidden(j) if *j > i => *j -= 1,
            _ => {}
        });
    }

    /// Runs the `callback` on the weights of the neural network in parallel, allowing it to modify weight values.
    pub fn update_weights(&mut self, callback: impl Fn(&NeuronLocation, &mut f32) + Sync) {
        for n in &mut self.input_layer {
            n.outputs
                .par_iter_mut()
                .for_each(|(loc, w)| callback(loc, w));
        }

        for n in &mut self.hidden_layers {
            n.outputs
                .par_iter_mut()
                .for_each(|(loc, w)| callback(loc, w));
        }
    }

    /// Runs the `callback` on the neurons of the neural network in parallel, allowing it to modify neuron values.
    pub fn mutate_neurons(&mut self, callback: impl Fn(&mut Neuron) + Sync) {
        self.input_layer.par_iter_mut().for_each(&callback);
        self.hidden_layers.par_iter_mut().for_each(&callback);
        self.output_layer.par_iter_mut().for_each(&callback);
    }

    /// Mutates the activation functions of the neurons in the neural network.
    pub fn mutate_activations(&mut self, rate: f32) {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        self.mutate_activations_with_reg(rate, &reg);
    }

    /// Mutates the activation functions of the neurons in the neural network, using a provided registry.
    pub fn mutate_activations_with_reg(&mut self, rate: f32, reg: &ActivationRegistry) {
        self.input_layer.par_iter_mut().for_each(|n| {
            let mut rng = rand::rng();
            if rng.random_bool(rate as f64) {
                n.mutate_activation(&reg.activations_in_scope(NeuronScope::INPUT), &mut rng);
            }
        });
        self.hidden_layers.par_iter_mut().for_each(|n| {
            let mut rng = rand::rng();
            if rng.random_bool(rate as f64) {
                n.mutate_activation(&reg.activations_in_scope(NeuronScope::HIDDEN), &mut rng);
            }
        });
        self.output_layer.par_iter_mut().for_each(|n| {
            let mut rng = rand::rng();
            if rng.random_bool(rate as f64) {
                n.mutate_activation(&reg.activations_in_scope(NeuronScope::OUTPUT), &mut rng);
            }
        });
    }

    /// Recounts inputs for all neurons in the network
    /// and removes any invalid connections.
    pub fn reset_input_counts(&mut self) {
        self.clear_input_counts();

        for i in 0..I {
            self.reset_inputs_for_neuron(NeuronLocation::Input(i));
        }

        for i in 0..self.hidden_layers.len() {
            self.reset_inputs_for_neuron(NeuronLocation::Hidden(i));
        }
    }

    fn reset_inputs_for_neuron(&mut self, loc: NeuronLocation) {
        let outputs = self[loc].outputs.keys().cloned().collect::<Vec<_>>();
        let outputs2 = outputs
            .into_iter()
            .filter(|&loc| {
                if !self.neuron_exists(loc) {
                    return false;
                }

                let target = &mut self[loc];
                target.input_count += 1;
                true
            })
            .collect::<HashSet<_>>();

        self[loc].outputs.retain(|loc, _| outputs2.contains(loc));
    }

    fn clear_input_counts(&mut self) {
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

    /// Iterates over the network and removes any hanging neurons in the hidden layer(s).
    pub fn prune_hanging_neurons(&mut self) {
        let mut i = 0;
        while i < self.hidden_layers.len() {
            let mut new_i = i + 1;
            if self.hidden_layers[i].input_count == 0 {
                // this saturating_sub is a code smell but it works and avoids some edge cases where indices can get messed up.
                new_i = new_i.saturating_sub(self.remove_neuron(NeuronLocation::Hidden(i)));
            }
            i = new_i;
        }
    }

    /// Uses DFS to find and remove all cycles in O(n+e) time.
    /// Expects [`prune_hanging_neurons`][NeuralNetwork::prune_hanging_neurons] to be called afterwards
    pub fn remove_cycles(&mut self) {
        let mut visited = HashMap::new();
        let mut edges_to_remove: HashSet<Connection> = HashSet::new();

        for i in 0..I {
            self.remove_cycles_dfs(
                &mut visited,
                &mut edges_to_remove,
                None,
                NeuronLocation::Input(i),
            );
        }

        // unattached cycles (will cause problems since they
        // never get deleted by input_count == 0)
        for i in 0..self.hidden_layers.len() {
            let loc = NeuronLocation::Hidden(i);
            if !visited.contains_key(&loc) {
                self.remove_cycles_dfs(&mut visited, &mut edges_to_remove, None, loc);
            }
        }

        for conn in edges_to_remove {
            // only doing raw here since we recalculate input counts and
            // prune hanging neurons later.
            self.remove_connection_raw(conn);
        }
    }

    // colored dfs
    fn remove_cycles_dfs(
        &mut self,
        visited: &mut HashMap<NeuronLocation, u8>,
        edges_to_remove: &mut HashSet<Connection>,
        prev: Option<NeuronLocation>,
        current: NeuronLocation,
    ) {
        if let Some(&existing) = visited.get(&current) {
            if existing == 0 {
                // part of current dfs - found a cycle
                // prev must exist here since visited would be empty on first call.
                let prev = prev.unwrap();
                if self[prev].outputs.contains_key(&current) {
                    edges_to_remove.insert(Connection {
                        from: prev,
                        to: current,
                    });
                }
            }

            // already fully visited, no need to check again
            return;
        }

        visited.insert(current, 0);

        let outputs = self[current].outputs.keys().cloned().collect::<Vec<_>>();
        for loc in outputs {
            self.remove_cycles_dfs(visited, edges_to_remove, Some(current), loc);
        }

        visited.insert(current, 1);
    }

    /// Performs just the mutations that modify the graph structure of the neural network,
    /// and not the internal mutations that only modify values such as activation functions, weights, and biases.
    pub fn perform_graph_mutations(
        &mut self,
        settings: &MutationSettings,
        rate: f32,
        rng: &mut impl rand::Rng,
    ) {
        // TODO maybe allow specifying probability
        // for each type of mutation
        if rng.random_bool(rate as f64) {
            // split connection
            if let Some(conn) = self.get_random_connection(settings.max_split_retries, rng) {
                self.split_connection(conn, rng);
            }
        }

        if rng.random_bool(rate as f64) {
            // add connection
            self.add_random_connection(settings.max_add_retries, rng);
        }

        if rng.random_bool(rate as f64) {
            // remove connection
            self.remove_random_connection(settings.max_remove_retries, rng);
        }
    }

    /// Performs just the mutations that modify internal values such as activation functions, weights, and biases,
    /// and not the graph mutations that modify the structure of the neural network.
    pub fn perform_internal_mutations(&mut self, settings: &MutationSettings, rate: f32) {
        self.mutate_activations(rate);
        self.mutate_weights(settings.weight_mutation_amount);
    }

    /// Same as [`mutate`][NeuralNetwork::mutate] but allows specifying a custom activation registry for activation mutations.
    pub fn mutate_with_reg(
        &mut self,
        settings: &MutationSettings,
        rate: f32,
        rng: &mut impl rand::Rng,
        reg: &ActivationRegistry,
    ) {
        self.perform_graph_mutations(settings, rate, rng);
        self.mutate_activations_with_reg(rate, reg);
        self.mutate_weights(settings.weight_mutation_amount);
    }

    /// Mutates all weights by a random amount up to `max_amount` in either direction.
    pub fn mutate_weights(&mut self, max_amount: f32) {
        self.update_weights(|_, w| {
            let mut rng = rand::rng();
            let amount = rng.random_range(-max_amount..max_amount);
            *w += amount;
        });
    }
}

impl<const I: usize, const O: usize> Index<NeuronLocation> for NeuralNetwork<I, O> {
    type Output = Neuron;

    fn index(&self, loc: NeuronLocation) -> &Self::Output {
        match loc {
            NeuronLocation::Input(i) => &self.input_layer[i],
            NeuronLocation::Hidden(i) => &self.hidden_layers[i],
            NeuronLocation::Output(i) => &self.output_layer[i],
        }
    }
}

impl<const I: usize, const O: usize> GenerateRandom for NeuralNetwork<I, O> {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self::new(rng)
    }
}

impl<const I: usize, const O: usize> IndexMut<NeuronLocation> for NeuralNetwork<I, O> {
    fn index_mut(&mut self, loc: NeuronLocation) -> &mut Self::Output {
        match loc {
            NeuronLocation::Input(i) => &mut self.input_layer[i],
            NeuronLocation::Hidden(i) => &mut self.hidden_layers[i],
            NeuronLocation::Output(i) => &mut self.output_layer[i],
        }
    }
}

/// The mutation settings for [`NeuralNetwork`].
/// Does not affect [`NeuralNetwork::mutate`], only [`NeuralNetwork::divide`] and [`NeuralNetwork::crossover`].
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MutationSettings {
    /// The chance of each mutation type to occur.
    pub mutation_rate: f32,

    /// The maximum amount that the weights will be mutated by in one mutation pass.
    pub weight_mutation_amount: f32,

    /// The maximum amount that biases will be mutated by in one mutation pass.
    pub bias_mutation_amount: f32,

    /// The maximum number of retries for adding connections.
    pub max_add_retries: usize,

    /// The maximum number of retries for removing connections.
    pub max_remove_retries: usize,

    /// The maximum number of retries for splitting connections.
    pub max_split_retries: usize,
}

impl Default for MutationSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.01,
            weight_mutation_amount: 0.5,
            bias_mutation_amount: 0.5,
            max_add_retries: 10,
            max_remove_retries: 10,
            max_split_retries: 10,
        }
    }
}

impl<const I: usize, const O: usize> RandomlyMutable for NeuralNetwork<I, O> {
    type Context = MutationSettings;

    fn mutate(&mut self, settings: &MutationSettings, rate: f32, rng: &mut impl Rng) {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        self.mutate_with_reg(settings, rate, rng, &reg);
    }
}

/// The settings used for [`NeuralNetwork`] reproduction.
#[derive(Debug, Clone, PartialEq)]
pub struct ReproductionSettings {
    /// The mutation settings to use during reproduction.
    pub mutation: MutationSettings,

    /// The number of times to apply mutation during reproduction.
    pub mutation_passes: usize,
}

impl Default for ReproductionSettings {
    fn default() -> Self {
        Self {
            mutation: MutationSettings::default(),
            mutation_passes: 3,
        }
    }
}

impl<const I: usize, const O: usize> Mitosis for NeuralNetwork<I, O> {
    type Context = ReproductionSettings;

    fn divide(
        &self,
        settings: &ReproductionSettings,
        rate: f32,
        rng: &mut impl prelude::Rng,
    ) -> Self {
        let mut child = self.clone();

        for _ in 0..settings.mutation_passes {
            child.mutate(&settings.mutation, rate, rng);
        }

        child
    }
}

/// The settings used for [`NeuralNetwork`] crossover.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CrossoverSettings {
    /// The reproduction settings to use during crossover, which will be applied to the child after crossover.
    pub repr: ReproductionSettings,
    // TODO other crossover settings.
}

impl<const I: usize, const O: usize> Crossover for NeuralNetwork<I, O> {
    type Context = CrossoverSettings;

    fn crossover(
        &self,
        other: &Self,
        settings: &CrossoverSettings,
        rate: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        // merge (temporarily breaking invariants) and then resolve invariants.
        let mut child = NeuralNetwork {
            input_layer: self.input_layer.clone(),
            hidden_layers: vec![],
            output_layer: self.output_layer.clone(),
        };

        for i in 0..I {
            if rng.random_bool(0.5) {
                child.input_layer[i] = other.input_layer[i].clone();
            }
        }

        for i in 0..O {
            if rng.random_bool(0.5) {
                child.output_layer[i] = other.output_layer[i].clone();
            }
        }

        let larger;
        let smaller;
        if self.hidden_layers.len() >= other.hidden_layers.len() {
            larger = &self.hidden_layers;
            smaller = &other.hidden_layers;
        } else {
            larger = &other.hidden_layers;
            smaller = &self.hidden_layers;
        }

        for i in 0..larger.len() {
            if i < smaller.len() {
                if rng.random_bool(0.5) {
                    child.hidden_layers.push(smaller[i].clone());
                } else {
                    child.hidden_layers.push(larger[i].clone());
                }
                continue;
            }

            // larger is the only one with spare neurons, add them.
            child.hidden_layers.push(larger[i].clone());
        }

        // resolve invariants
        child.remove_cycles();
        child.reset_input_counts();
        child.prune_hanging_neurons();

        for _ in 0..settings.repr.mutation_passes {
            child.mutate(&settings.repr.mutation, rate, rng);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    #[cfg_attr(feature = "serde", serde(with = "outputs_serde"))]
    pub outputs: HashMap<NeuronLocation, f32>,

    /// The initial value of the neuron.
    pub bias: f32,

    /// The activation function applied to the value before propagating to [`outputs`][Neuron::outputs].
    pub activation_fn: ActivationFn,
}

impl Neuron {
    /// Creates a new neuron with a specified activation function and outputs.
    pub fn new_with_activation(
        outputs: HashMap<NeuronLocation, f32>,
        activation_fn: ActivationFn,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            input_count: 0,
            outputs,
            bias: rng.random(),
            activation_fn,
        }
    }

    /// Creates a new neuron with the given output locations.
    /// Chooses a random activation function within the specified scope.
    pub fn new(
        outputs: HashMap<NeuronLocation, f32>,
        scope: NeuronScope,
        rng: &mut impl Rng,
    ) -> Self {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        let act = reg.random_activation_in_scope(scope, rng);

        Self::new_with_activation(outputs, act, rng)
    }

    /// Creates a new neuron with the given outputs.
    /// Takes a collection of activation functions and chooses a random one from them to use.
    pub fn new_with_activations(
        outputs: HashMap<NeuronLocation, f32>,
        activations: &[ActivationFn],
        rng: &mut impl Rng,
    ) -> Self {
        // TODO maybe Result instead.
        if activations.is_empty() {
            panic!("Empty activations list provided");
        }

        Self::new_with_activation(
            outputs,
            activations[rng.random_range(0..activations.len())].clone(),
            rng,
        )
    }

    /// Runs the [activation function][Neuron::activation_fn] on the given value and returns it.
    pub fn activate(&self, v: f32) -> f32 {
        self.activation_fn.func.activate(v)
    }

    /// Randomly mutates the specified weight with the rate.
    pub fn mutate_weight(
        &mut self,
        output: NeuronLocation,
        rate: f32,
        rng: &mut impl Rng,
    ) -> Option<f32> {
        if let Some(w) = self.outputs.get_mut(&output) {
            *w += rng.random_range(-rate..=rate);
            return Some(*w);
        }

        None
    }

    /// Get a random output location and weight.
    pub fn random_output(&self, rng: &mut impl Rng) -> (NeuronLocation, f32) {
        // will panic if outputs is empty
        let i = rng.random_range(0..self.outputs.len());
        let x = self.outputs.iter().nth(i).unwrap();
        (*x.0, *x.1)
    }

    pub(crate) fn downshift_outputs(&mut self, i: usize) {
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
            .retain(|loc, _| output_exists(*loc, hidden_len, output_len));
    }

    /// Replaces the activation function with a random one.
    pub fn mutate_activation(&mut self, activations: &[ActivationFn], rng: &mut impl Rng) {
        if activations.is_empty() {
            panic!("Empty activations list provided");
        }

        self.activation_fn = activations[rng.random_range(0..activations.len())].clone();
    }
}

/// A pseudo-pointer of sorts that is used for caching.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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

    /// Sets the inner index value without changing the layer.
    pub fn set_inner(&mut self, v: usize) {
        // there's gotta be a cleaner way of doing this
        match self {
            Self::Input(i) => *i = v,
            Self::Hidden(i) => *i = v,
            Self::Output(i) => *i = v,
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
    pub fn get(&self, loc: NeuronLocation) -> f32 {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].value.load(Ordering::SeqCst),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].value.load(Ordering::SeqCst),
            NeuronLocation::Output(i) => self.output_layer[i].value.load(Ordering::SeqCst),
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

        let hidden_layers = net.hidden_layers.par_iter().map(|n| n.into()).collect();

        let output_layer: Vec<_> = net.output_layer.par_iter().map(|n| n.into()).collect();
        let output_layer = output_layer.try_into().unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}
