use std::{fmt, sync::Arc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

/// A trait that represents an activation method.
pub trait Activation {
    /// The activation function.
    fn activate(&self, n: f32) -> f32;
}

impl<F: Fn(f32) -> f32> Activation for F {
    fn activate(&self, n: f32) -> f32 {
        (self)(n)
    }
}

/// An activation function object that implements [`fmt::Debug`] and is [`Send`]
#[derive(Clone)]
pub struct ActivationFn {
    /// The actual activation function.
    pub func: Arc<dyn Activation>,
    name: String,
}

impl fmt::Debug for ActivationFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)
    }
}

impl PartialEq for ActivationFn {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[cfg(feature = "serde")]
impl Serialize for ActivationFn {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.name)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for ActivationFn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let name = String::deserialize(deserializer)?;
        let activations = activation_fn! {
            sigmoid,
            relu,
            f32::tanh,
            linear_activation
        };

        for a in activations {
            if a.name == name {
                return Ok(a);
            }
        }

        // eventually will make an activation fn registry of sorts.
        panic!("Custom activation functions currently not supported.") // TODO return error instead of raw panic
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
pub fn linear_activation(n: f32) -> f32 {
    n
}