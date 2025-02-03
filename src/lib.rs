//! # neat
//! A crate implementing NeuroEvolution of Augmenting Topologies (NEAT).
//! 
//! The goal is to provide a simple-to-use, very dynamic [`NeuralNetwork`] type that
//! integrates directly into the [`genetic-rs`](https://crates.io/crates/genetic-rs) ecosystem.
//! 
//! Look at the README, docs, or examples to learn how to use this crate.

#![warn(missing_docs)]

/// Contains the types surrounding activation functions.
pub mod activation;

/// Contains the [`NeuralNetwork`] and related types.
pub mod neuralnet;

pub use neuralnet::*;

pub use genetic_rs::{self, prelude::*};

#[cfg(test)]
mod tests;
