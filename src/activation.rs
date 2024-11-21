#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use bitflags::bitflags;
use lazy_static::lazy_static;
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use crate::NeuronLocation;

/// Creates an [`ActivationFn`] object from a function
#[macro_export]
macro_rules! activation_fn {
    ($F: path) => {
        ActivationFn::new(std::sync::Arc::new($F), ActivationScope::default(), stringify!($F).into())
    };

    ($F: path, $S: expr) => {
        ActivationFn::new(std::sync::Arc::new($F), $S, stringify!($F).into())
    };

    {$($F: path),*} => {
        [$(activation_fn!($F)),*]
    };

    {$($F: path => $S: expr),*} => {
        [$(activation_fn!($F, $S)),*]
    }
}

lazy_static! {
    /// A static activation registry for use in deserialization.
    pub(crate) static ref ACTIVATION_REGISTRY: Arc<RwLock<ActivationRegistry>> = Arc::new(RwLock::new(ActivationRegistry::default()));
}

/// Register an activation function to the registry.
pub fn register_activation(act: ActivationFn) {
    let mut reg = ACTIVATION_REGISTRY.write().unwrap();
    reg.register(act);
}

/// Registers multiple activation functions to the registry at once.
pub fn batch_register_activation(acts: impl IntoIterator<Item = ActivationFn>) {
    let mut reg = ACTIVATION_REGISTRY.write().unwrap();
    reg.batch_register(acts);
}

/// A registry of the different possible activation functions.
pub struct ActivationRegistry {
    /// The currently-registered activation functions.
    pub fns: HashMap<String, ActivationFn>,
}

impl ActivationRegistry {
    /// Registers an activation function.
    pub fn register(&mut self, activation: ActivationFn) {
        self.fns.insert(activation.name.clone(), activation);
    }

    /// Registers multiple activation functions at once.
    pub fn batch_register(&mut self, activations: impl IntoIterator<Item = ActivationFn>) {
        for act in activations {
            self.register(act);
        }
    }

    /// Gets a Vec of all the activation functions registered. Unless you need an owned value, use [fns][ActivationRegistry::fns].values() instead.
    pub fn activations(&self) -> Vec<ActivationFn> {
        self.fns.values().cloned().collect()
    }

    /// Gets all activation functions that are valid for a scope.
    pub fn activations_in_scope(&self, scope: ActivationScope) -> Vec<ActivationFn> {
        let acts = self.activations();

        acts.into_iter()
            .filter(|a| !a.scope.contains(ActivationScope::NONE) && a.scope.contains(scope))
            .collect()
    }
}

impl Default for ActivationRegistry {
    fn default() -> Self {
        let mut s = Self {
            fns: HashMap::new(),
        };

        // TODO add a way to disable this
        s.batch_register(activation_fn! {
            sigmoid => ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            relu => ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            linear_activation => ActivationScope::INPUT | ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            f32::tanh => ActivationScope::HIDDEN | ActivationScope::OUTPUT
        });

        s
    }
}

bitflags! {
    /// Specifies where an activation function can occur
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct ActivationScope: u8 {
        /// Whether the activation can be applied to the input layer.
        const INPUT = 0b001;

        /// Whether the activation can be applied to the hidden layer.
        const HIDDEN = 0b010;

        /// Whether the activation can be applied to the output layer.
        const OUTPUT = 0b100;

        /// The activation function will not be randomly placed anywhere
        const NONE = 0b000;
    }
}

impl Default for ActivationScope {
    fn default() -> Self {
        Self::HIDDEN
    }
}

impl From<&NeuronLocation> for ActivationScope {
    fn from(value: &NeuronLocation) -> Self {
        match value {
            NeuronLocation::Input(_) => Self::INPUT,
            NeuronLocation::Hidden(_) => Self::HIDDEN,
            NeuronLocation::Output(_) => Self::OUTPUT,
        }
    }
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
    pub func: Arc<dyn Activation + Send + Sync>,

    /// The scope defining where the activation function can appear.
    pub scope: ActivationScope,
    pub(crate) name: String,
}

impl ActivationFn {
    /// Creates a new ActivationFn object.
    pub fn new(
        func: Arc<dyn Activation + Send + Sync>,
        scope: ActivationScope,
        name: String,
    ) -> Self {
        Self { func, name, scope }
    }
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

        let reg = ACTIVATION_REGISTRY.read().unwrap();

        let f = reg.fns.get(&name);

        if f.is_none() {
            panic!("Activation function {name} not found");
        }

        Ok(f.unwrap().clone())
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