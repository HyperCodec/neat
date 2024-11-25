pub mod activation;
pub mod neuralnet;

pub use activation::{*, fns::*};
pub use neuralnet::*;

pub use genetic_rs::{self, prelude::*};

#[cfg(test)]
mod major_tests;