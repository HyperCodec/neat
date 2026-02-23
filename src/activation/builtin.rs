/// The sigmoid activation function. Scales all values nonlinearly to the range (0, 1).
pub fn sigmoid(n: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-n))
}

/// The ReLU activation function. Equal to `n.max(0)`
pub fn relu(n: f32) -> f32 {
    n.max(0.)
}

/// Activation function that does nothing.
pub fn linear_activation(n: f32) -> f32 {
    n
}
