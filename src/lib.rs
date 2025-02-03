#![warn(missing_docs)]

pub mod activation;
pub mod neuralnet;

pub use neuralnet::*;

pub use genetic_rs::{self, prelude::*};

#[cfg(test)]
mod tests;
