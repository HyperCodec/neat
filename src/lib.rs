//! A simple crate that implements the Neuroevolution Augmenting Topologies algorithm using [genetic-rs](https://crates.io/crates/genetic-rs)
//! ### Feature Roadmap:
//! - [x] base (single-core) crate
//! - [x] rayon
//! - [ ] crossover
//! 
//! You can get started by 

#![warn(missing_docs)]

/// A module containing the [`NeuralNetworkTopology`] struct. This is what you want to use in the DNA of your agent, as it is the thing that goes through nextgens and suppors mutation.
pub mod topology;

/// A module containing the main [`NeuralNetwork`] struct.
/// This has state/cache and will run the predictions. Make sure to run [`NeuralNetwork::flush_state`] between uses of [`NeuralNetwork::predict`].
pub mod runnable;

pub use genetic_rs::prelude::*;
pub use topology::*;
pub use runnable::*;