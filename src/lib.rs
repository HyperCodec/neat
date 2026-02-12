#![doc = include_str!("../README.md")]

#![warn(missing_docs)]

/// Contains the types surrounding activation functions.
pub mod activation;

/// Contains the [`NeuralNetwork`] and related types.
pub mod neuralnet;

pub use neuralnet::*;

pub use genetic_rs::{self, prelude::*};

/// A trait for getting the index of the maximum element.
pub trait MaxIndex {
    /// Returns the index of the maximum element.
    fn max_index(self) -> Option<usize>;
}

impl<T: PartialOrd, I: Iterator<Item = T>> MaxIndex for I {
    fn max_index(self) -> Option<usize> {
        // enumerate now so we don't accidentally
        // skip the index of the first element
        let mut iter = self.enumerate();
        
        let mut max_i = 0;

        let first = iter.next();
        if first.is_none() {
            return None;
        }

        let mut max_v = first.unwrap().1;

        for (i, v) in iter {
            if v > max_v {
                max_v = v;
                max_i = i;
            }
        }

        Some(max_i)
    }
}

#[cfg(test)]
mod tests;