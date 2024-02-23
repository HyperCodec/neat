//! A simple crate that implements the Neuroevolution Augmenting Topologies algorithm using [genetic-rs](https://crates.io/crates/genetic-rs)
//! ### Feature Roadmap:
//! - [x] base (single-core) crate
//! - [x] rayon
//! - [x] serde
//! - [x] crossover
//!
//! You can get started by looking at [genetic-rs docs](https://docs.rs/genetic-rs) and checking the examples for this crate.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// A module containing the [`NeuralNetworkTopology`] struct. This is what you want to use in the DNA of your agent, as it is the thing that goes through nextgens and suppors mutation.
pub mod topology;

/// A module containing the main [`NeuralNetwork`] struct.
/// This has state/cache and will run the predictions. Make sure to run [`NeuralNetwork::flush_state`] between uses of [`NeuralNetwork::predict`].
pub mod runnable;

pub use genetic_rs::prelude::*;
pub use runnable::*;
pub use topology::*;

#[cfg(feature = "serde")]
pub use nnt_serde::*;
