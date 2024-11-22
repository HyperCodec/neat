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
